import torch
import torch.nn as nn
import numpy as np
import attention
from torch.nn.utils.rnn import pad_sequence

class generator(nn.Module):
    def __init__(self, config, word_embed=None):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_idx)
        if word_embed is not None:
            self.embedding.from_pretrained(torch.FloatTensor(word_embed), freeze=False)

        # L1 LSTM
        self.l1_lstm_cell = nn.LSTMCell(config.embed_size, config.g_hidden_size, bias=True)
        self.l1_img_to_embed = nn.Linear(3 * config.image_hidden, config.embed_size)
        # convert lstm hidden to predicted vocab
        self.l1_output_dropout = nn.Dropout(config.dropout)
        self.l1_output_linear = nn.Linear(config.g_hidden_size, config.vocab_size)

        # L2 LSTM
        self.l2_lstm_cell = nn.LSTMCell(config.embed_size, config.g_hidden_size, bias=True)
        self.l2_img_to_embed = nn.Linear(3 * config.image_hidden, config.embed_size)
        # convert lstm hidden to predicted vocab
        self.l2_output_dropout = nn.Dropout(config.dropout)
        self.l2_output_linear = nn.Linear(config.g_hidden_size, config.vocab_size)
        # L2 attention
        self.l2_cur_hidden_linear = nn.Linear(config.g_hidden_size, config.g_hidden_size)
        self.l2_prev_hidden_linear = nn.Linear(config.g_hidden_size, config.g_hidden_size)
        self.l2_attend_linear = nn.Linear(config.g_hidden_size, 1)
        self.gate_linear = nn.Linear(config.g_hidden_size * 2, 1)

        # used for multinomial sampling
        self.softmax = nn.Softmax(dim=1)

        # convert img_feature to hidden
        self.relu = nn.ReLU()

        # Attention part
        self.atten_layer = attention.MultiHeadAttention(config)

    def forward(self, entity_feature, neighbor_feature_list, img_feature,
                target_sent1=None, target_sent_len1=None, target_sent2=None, target_sent_len2=None, fix_seq_len=-1):
        """
        Add attention for target entity to neighbors
        :param entity_feature: entity's extracted feature, [batch_size, img_hidden]
        :param neighbor_feature_list: neighbors feature, List[ tensor(neighbor_num, img_hidden) ]
        :param img_feature: image's feature, [batch_size, img_hidden]
        :param target_sent: target sentence for teacher forcing, [batch_size, seq_len]
                            assume [START, sentence, END], and has already converted to LongTensor
        :param target_sent_len: record each sentences' len, [batch_size]
        :param fix_seq_len: we fix input until index = fix_seq_len , if t > fix_seq_len,
                            use multinomial sampling rather than groudtruth
        :return: pred_words_probs: [batch_size, seq_len - 1, vocab], no START
                                   each element represents the predict probability in each t
                 sampled_sent: in training phase, it is [batch_size, seq_len], represent the teacher forcing sent.
                               in infering phase, it is [batch_size, seq_len]
                 sampled_sent_len: [batch_size], represent the lens of each example in sampled_sent
        """
        # Attention based image feature
        final_img_feature = self.generate_final_feature(entity_feature, neighbor_feature_list, img_feature)
        l1_pred_words_probs, l1_sampled_sent, l1_sampled_sent_len, l1_hiddens = \
            self.first_layer_forward(final_img_feature, target_sent1, target_sent_len1, fix_seq_len)

        if target_sent2 is not None and target_sent_len2 is not None:
            attend_mask = self.build_attend_mask(target_sent_len1)  # [B, seq_len - 1]
            l2_pred_words_probs, l2_sampled_sent, l2_sampled_sent_len = \
                self.second_layer_forward(final_img_feature, target_sent2, target_sent_len2, l1_hiddens, attend_mask, fix_seq_len)
        else:
            l2_pred_words_probs, l2_sampled_sent, l2_sampled_sent_len = None, None, None

        return l1_pred_words_probs, l1_sampled_sent, l1_sampled_sent_len, l2_pred_words_probs, l2_sampled_sent, l2_sampled_sent_len

    def first_layer_forward(self, img_feature, target_sent, target_sent_len, fix_seq_len=-1):
        """
        :param img_feature: Attention based img feature, [batch, 3 * img_hidden]
        :param target_sent: easier target sentence for teacher forcing, [batch_size, seq_len]
                            assume [START, sentence, END], and has already converted to LongTensor
        :param target_sent_len: record each sentences' len, [batch_size]
        :param fix_seq_len: we fix input until index = fix_seq_len , if t > fix_seq_len,
                            use multinomial sampling rather than groudtruth
        :return: pred_words_probs: [batch_size, seq_len - 1, vocab], no START
                                   each element represents the predict probability in each t
                 sampled_sent: in training phase, it is [batch_size, seq_len], represent the teacher forcing sent.
                               in infering phase, it is [batch_size, seq_len]
                 sampled_sent_len: [batch_size], represent the lens of each example in sampled_sent
                 hiddens: [batch_size, seq_len - 1, hidden], no START, each element represent the hidden states in each t
        """
        batch_size = img_feature.size(0)
        pred_words_probs = []
        hiddens = []

        # in the first timestep, we feed in the image feature
        img_embed_feature = torch.tanh(self.l1_img_to_embed(img_feature))
        h, c = self.l1_lstm_cell(img_embed_feature)

        # Teacher forcing
        if target_sent is not None and fix_seq_len == -1:
            # here is training phase, where provide target sentence
            # use sampled_sent because we will change the words
            sampled_sent = target_sent.clone()
            seq_len = target_sent.size(1)
            sampled_sent_embed = self.embedding(sampled_sent)   # [batch_size, seq_len, embed_size]

            # iterate whole sequence
            prev_sampled = None
            for t in range(seq_len - 1):    # seq_len - 1 means ignore END
                t_sent_embed = sampled_sent_embed[:, t, :] if prev_sampled is None else self.embedding(prev_sampled)
                t_sent_embed = t_sent_embed.reshape(batch_size, -1)
                h, c = self.l1_lstm_cell(t_sent_embed, (h, c))
                t_pred_word = self.l1_output_linear(self.l1_output_dropout(h))  # [batch_size, vocab]
                pred_words_probs.append(t_pred_word)
                hiddens.append(h)

            sampled_sent_len = target_sent_len.clone().view(-1)  # output_len = input_len
            # [batch_size, seq_len-1, vocab_size]
            pred_words_probs = torch.stack(pred_words_probs).permute(1, 0, 2).to(self.config.device)
            hiddens = torch.stack(hiddens).permute(1, 0, 2).to(self.config.device)  # [batch_size, seq_len-1, hidden_size]

        return pred_words_probs, sampled_sent, sampled_sent_len, hiddens

    def second_layer_forward(self, img_feature, target_sent, target_sent_len, l1_hidden, attend_mask, fix_seq_len=-1):
        """
        :param img_feature: Attention based img feature, [batch, 3 * img_hidden]
        :param target_sent: easier target sentence for teacher forcing, [batch_size, seq_len]
                            assume [START, sentence, END], and has already converted to LongTensor
        :param target_sent_len: record each sentences' len, [batch_size]
        :param l1_hidden: [batch_size, seq_len - 1, hidden]
        :param fix_seq_len: we fix input until index = fix_seq_len , if t > fix_seq_len,
                            use multinomial sampling rather than groudtruth
        :return: pred_words_probs: [batch_size, seq_len - 1, vocab], no START
                                   each element represents the predict probability in each t
                 sampled_sent: in training phase, it is [batch_size, seq_len], represent the teacher forcing sent.
                               in infering phase, it is [batch_size, seq_len]
                 sampled_sent_len: [batch_size], represent the lens of each example in sampled_sent
        """
        batch_size = img_feature.size(0)
        pred_words_probs = []

        # in the first timestep, we feed in the image feature
        img_embed_feature = torch.tanh(self.l2_img_to_embed(img_feature))
        h, c = self.l2_lstm_cell(img_embed_feature)

        # Teacher forcing
        if target_sent is not None and fix_seq_len == -1:
            # here is training phase, where provide target sentence
            # use sampled_sent because we will change the words
            sampled_sent = target_sent.clone()
            seq_len = target_sent.size(1)
            sampled_sent_embed = self.embedding(sampled_sent)  # [batch_size, seq_len, embed_size]

            # iterate whole sequence
            prev_sampled = None
            for t in range(seq_len - 1):  # seq_len - 1 means ignore END
                t_sent_embed = sampled_sent_embed[:, t, :] if prev_sampled is None else self.embedding(prev_sampled)
                t_sent_embed = t_sent_embed.reshape(batch_size, -1)
                c = self.attention_module(l1_hidden, h, c, attend_mask)
                h, c = self.l2_lstm_cell(t_sent_embed, (h, c))
                t_pred_word = self.l2_output_linear(self.l2_output_dropout(h))  # [batch_size, vocab]
                pred_words_probs.append(t_pred_word)

            sampled_sent_len = target_sent_len.clone().view(-1)  # output_len = input_len
            # [batch_size, seq_len-1, vocab_size]
            pred_words_probs = torch.stack(pred_words_probs).permute(1, 0, 2).to(self.config.device)

        return pred_words_probs, sampled_sent, sampled_sent_len

    def attention_module(self, l1_hidden, h, c, attend_mask):
        """
        :param l1_hidden: [batch, seq_len - 1, hidden]
        :param h: [batch, hidden]
        :param c: [batch, hidden]
        :return:
        """
        batch_size, seq_len, hidden_size = l1_hidden.shape

        # Get attend c
        h_expand = h.unsqueeze(1).repeat(1, seq_len, 1)    # [B, S, H]
        attend = torch.tanh(self.l2_cur_hidden_linear(h_expand) + self.l2_prev_hidden_linear(l1_hidden))    # [B, S, H]
        attend = self.l2_attend_linear(attend).squeeze(2)    # [B, S]
        attend = attend.masked_fill_(attend_mask, float('-inf'))
        attend = torch.softmax(attend, dim=1).unsqueeze(1) # [B, 1, S]
        attend_c = torch.bmm(attend, l1_hidden).squeeze(1)   # [B, H]
        
        # first compute the gate then compute final c
        gate = torch.sigmoid(self.gate_linear(torch.cat([h, attend_c], dim=1))) # [B, 1]
        new_c = torch.mul(gate, c) + torch.mul(1 - gate, attend_c)
        return new_c

    def build_attend_mask(self, target_sent_len, max_seq=None):
        """
        :param target_sent_len: record each sentences' len, [batch_size]
        :return:
        """
        target_sent_len = target_sent_len.view(-1)
        target_sent_len = target_sent_len - 1   # ignore START
        batch_size = len(target_sent_len)
        max_seq_len = torch.max(target_sent_len).item() if max_seq == None else max_seq

        sent_len_expand = target_sent_len.unsqueeze(1).expand((batch_size, max_seq_len))    # [batch, seq_len - 1]
        seq_range = torch.arange(0, max_seq_len).long().to(self.config.device)
        seq_range_expand = seq_range.unsqueeze(0).expand((batch_size, max_seq_len))
        atten_mask = sent_len_expand <= seq_range_expand
        return atten_mask

    def greedy_generate(self, entity_feature, neighbor_feature_list, img_feature):
        """
        We use greedy policy to generate the sentence
        Each timestep t we choose the most possible token
        :param entity_feature: entity's extracted feature, [batch_size, img_hidden]
        :param neighbor_feature_list: neighbors feature, List[ tensor(neighbor_num, img_hidden) ]
        :param img_feature: image's feature, [batch_size, img_hidden]
        :return: l1_sent: List[List[]], each element (List) means i-th generated sentence, without END and START
        """
        # in the first timestep, we feed in the image feature
        final_img_feature = self.generate_final_feature(entity_feature, neighbor_feature_list, img_feature)
        l1_sent, l1_sent_len, hiddens = self.first_layer_greedy_generate(final_img_feature)
        attend_mask = self.build_attend_mask(l1_sent_len, max_seq=self.config.max_sequence-1)
        l2_sent = self.second_layer_greedy_generate(final_img_feature, hiddens, attend_mask)
        return l1_sent, l2_sent

    def first_layer_greedy_generate(self, final_img_feature):
        batch_size = final_img_feature.size(0)
        pred_sents = [torch.LongTensor((batch_size)).fill_(self.config.start_idx).to(self.config.device)]
        hiddens = []

        img_embed_feature = torch.tanh(self.l1_img_to_embed(final_img_feature))
        h, c = self.l1_lstm_cell(img_embed_feature)

        for t in range(self.config.max_sequence - 1):
            input_seq = pred_sents[-1]
            input_seq_embed = self.embedding(input_seq)
            h, c = self.l1_lstm_cell(input_seq_embed, (h, c))
            hiddens.append(h)
            t_pred_word = self.l1_output_linear(self.l1_output_dropout(h))

            # use greedy policy to choose
            t_max_word = torch.max(t_pred_word, dim=1)[1]
            pred_sents.append(t_max_word.reshape(batch_size))

        final_sent = []
        for t in range(1, self.config.max_sequence):
            words = pred_sents[t]
            if t == 1:
                final_sent = [[word.item()] for word in words]
            else:
                for idx, word in enumerate(words):
                    if final_sent[idx][-1] == self.config.end_idx:
                        continue
                    final_sent[idx].append(word.item())
        final_sent = [sent[:-1] if sent[-1] == self.config.end_idx else sent for sent in final_sent]
        final_sent_len = torch.LongTensor([len(sent) + 1 for sent in final_sent]).to(self.config.device)    # +1 because add last word to END
        hiddens = torch.stack(hiddens).permute(1, 0, 2).to(self.config.device)  # [batch_size, seq_len-1, hidden_size]
        return final_sent, final_sent_len, hiddens

    def second_layer_greedy_generate(self, final_img_feature, l1_hidden, attend_mask):
        batch_size = final_img_feature.size(0)
        pred_sents = [torch.LongTensor((batch_size)).fill_(self.config.start_idx).to(self.config.device)]

        img_embed_feature = torch.tanh(self.l2_img_to_embed(final_img_feature))
        h, c = self.l2_lstm_cell(img_embed_feature)

        for t in range(self.config.max_sequence - 1):
            input_seq = pred_sents[-1]
            input_seq_embed = self.embedding(input_seq)
            c = self.attention_module(l1_hidden, h, c, attend_mask)
            h, c = self.l2_lstm_cell(input_seq_embed, (h, c))
            t_pred_word = self.l2_output_linear(self.l2_output_dropout(h))

            # use greedy policy to choose
            t_max_word = torch.max(t_pred_word, dim=1)[1]
            pred_sents.append(t_max_word.reshape(batch_size))

        final_sent = []
        for t in range(1, self.config.max_sequence):
            words = pred_sents[t]
            if t == 1:
                final_sent = [[word.item()] for word in words]
            else:
                for idx, word in enumerate(words):
                    if final_sent[idx][-1] == self.config.end_idx:
                        continue
                    final_sent[idx].append(word.item())
        final_sent = [sent[:-1] if sent[-1] == self.config.end_idx else sent for sent in final_sent]
        return final_sent

    def generate_final_feature(self, entity_feature, neighbor_feature_list, img_feature):
        """
        Add attention for target entity to neighbors
        :param entity_feature: entity's extracted feature, [batch_size, img_hidden]
        :param neighbor_feature_list: neighbors feature, List[ tensor(neighbor_num, img_hidden) ]
        :param img_feature: image's feature, [batch_size, img_hidden]

        :return final_feature: concat three types of the feature, [batch, 3 * img_hidden]
        """
        batch_size = img_feature.size(0)
        img_hidden = img_feature.size(1)

        # First to pad the neighbor_feature
        neighbor_feature = pad_sequence(neighbor_feature_list, batch_first=True)  # [batch, max_n_num, img_hidden]
        mask_n_num = neighbor_feature.size(1)

        # generate attention mask
        neighbor_num = torch.LongTensor([neighbor.size(0) for neighbor in neighbor_feature_list]).to(self.config.device)
        neighbor_num_expand = neighbor_num.unsqueeze(1).expand(batch_size, mask_n_num)
        seq_range = torch.arange(0, mask_n_num).long().to(self.config.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, mask_n_num)
        atten_mask = neighbor_num_expand <= seq_range_expand
        atten_mask = atten_mask.unsqueeze(1)

        # generate attended_neighbor feature
        attend_neighbor_feature, _ = self.atten_layer(key=neighbor_feature, value=neighbor_feature,
                                                   query=entity_feature.view(batch_size, 1, img_hidden),
                                                   attn_mask=atten_mask)    # [batch, 1, img_hidden]

        # concat three types of feature
        attend_neighbor_feature = attend_neighbor_feature.view(batch_size, img_hidden)
        final_feature = torch.cat([entity_feature, attend_neighbor_feature, img_feature], dim=-1)   # [batch, 3 * img_hidden]
        return final_feature