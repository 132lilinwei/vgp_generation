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

        self.lstm_cell = nn.LSTMCell(config.embed_size, config.g_hidden_size, bias=True)

        # used for multinomial sampling
        self.softmax = nn.Softmax(dim=1)

        # convert img_feature to hidden
        self.relu = nn.ReLU()
        self.img_to_embed = nn.Linear(3 * config.image_hidden, config.embed_size)

        # convert lstm hidden to predicted vocab
        self.output_dropout = nn.Dropout(config.dropout)
        self.output_linear = nn.Linear(config.g_hidden_size, config.vocab_size)

        # Attention part
        self.atten_layer = attention.MultiHeadAttention(config)
        self.atten_layer2 = attention.SoftDotAttention(config)

    def forward(self, entity_feature, neighbor_feature_list, img_feature,
                target_sent=None, target_sent_len=None, fix_seq_len=-1):
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

        batch_size = img_feature.size(0)
        pred_words_probs = []

        # in the first timestep, we feed in the image feature
        # here convert image_hidden to embed_size
        final_img_feature = self.generate_final_feature(entity_feature, neighbor_feature_list, img_feature)
        img_embed_feature = torch.tanh(self.img_to_embed(final_img_feature))
        h, c = self.lstm_cell(img_embed_feature)

        # then in the training phase, we use target_sent for teacher forcing
        if (target_sent is not None) and fix_seq_len == -1:
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
                h, c = self.lstm_cell(t_sent_embed, (h, c))
                t_pred_word = self.output_linear(self.output_dropout(h))  # [batch_size, vocab]
                pred_words_probs.append(t_pred_word)

            sampled_sent_len = target_sent_len.clone().view(-1)  # output_len = input_len
            # [batch_size, seq_len-1, vocab_size]
            pred_words_probs = torch.stack(pred_words_probs).permute(1, 0, 2).to(self.config.device)

        else:
            # here is inference time to generate caption
            still_gen_mask = np.ones(batch_size, dtype=np.int)
            np_sent_len = np.ones(batch_size, dtype=np.int64) * self.config.max_sequence

            # first add START in the begining
            sampled_sent = [torch.LongTensor((batch_size)).fill_(self.config.start_idx).to(self.config.device)]
            for t in range(0, self.config.max_sequence - 1):
                input_seq = sampled_sent[-1]

                input_seq_embed = self.embedding(input_seq)
                h, c = self.lstm_cell(input_seq_embed, (h, c))
                t_pred_word = self.output_linear(self.output_dropout(h))
                pred_words_probs.append(t_pred_word)

                if t >= fix_seq_len:
                    # if t step is larger than fix_seq_len, we should use multinomial sample words
                    # will use target_sent_len to mask padding case.
                    sampled = torch.multinomial(self.softmax(t_pred_word), 1)  # [batch_size, 1]
                    sampled_sent.append(sampled.reshape(batch_size))
                else:
                    sampled_sent.append(target_sent[:, t + 1])

                end_next = sampled_sent[-1].reshape(batch_size) == self.config.end_idx
                end_next = end_next.detach().cpu().numpy() == 1
                still_gen_mask = np.logical_and(still_gen_mask, 1 - end_next)
                if np.sum(end_next) > 0:
                    np_sent_len[end_next] = np.minimum(np_sent_len[end_next],
                                                       t + 1 + 1)  # +1: index to length, +1: output for next word
                if np.sum(still_gen_mask) == 0:
                    break

            # [batch_size, seq_len]
            sampled_sent = torch.stack(sampled_sent).permute(1, 0).to(self.config.device)
            sampled_sent_len = torch.tensor(np_sent_len, dtype=torch.int64).to(self.config.device).view(-1)
            # [batch_size, seq_len-1, vocab_size]
            pred_words_probs = torch.stack(pred_words_probs).permute(1, 0, 2).to(self.config.device)

        return pred_words_probs, sampled_sent, sampled_sent_len

    def greedy_generate(self, entity_feature, neighbor_feature_list, img_feature,eid = None):
        """
        We use greedy policy to generate the sentence
        Each timestep t we choose the most possible token
        :param entity_feature: entity's extracted feature, [batch_size, img_hidden]
        :param neighbor_feature_list: neighbors feature, List[ tensor(neighbor_num, img_hidden) ]
        :param img_feature: image's feature, [batch_size, img_hidden]
        :return: final_sent: List[List[]], each element (List) means i-th generated sentence, without END and START
        """
        batch_size = img_feature.size(0)
        pred_sents = [torch.LongTensor((batch_size)).fill_(self.config.start_idx).to(self.config.device)]

        # in the first timestep, we feed in the image feature
        # here convert image_hidden to embed_size
        final_img_feature = self.generate_final_feature(entity_feature, neighbor_feature_list, img_feature,eid)
        img_embed_feature = torch.tanh(self.img_to_embed(final_img_feature))
        h, c = self.lstm_cell(img_embed_feature)

        for t in range(self.config.max_sequence - 1):
            input_seq = pred_sents[-1]

            input_seq_embed = self.embedding(input_seq)
            h, c = self.lstm_cell(input_seq_embed, (h, c))
            t_pred_word = self.output_linear(self.output_dropout(h))

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


    def beam_sample(self, entity_feature, neighbor_feature_list, img_feature, beam_size = 8):
        """
        We use greedy policy to generate the sentence
        Each timestep t we choose the most possible token
        :param entity_feature: entity's extracted feature, [batch_size, img_hidden]
        :param neighbor_feature_list: neighbors feature, List[ tensor(neighbor_num, img_hidden) ]
        :param img_feature: image's feature, [batch_size, img_hidden]
        :return: final_sent: List[List[]], each element (List) means i-th generated sentence, without END and START
                 prob: List[Int], corresponding probabilities

        """
        batch_size = img_feature.size(0)

        # in the first timestep, we feed in the image feature
        # here convert image_hidden to embed_size
        final_img_feature = self.generate_final_feature(entity_feature, neighbor_feature_list, img_feature)
        img_embed_feature = torch.tanh(self.img_to_embed(final_img_feature))

        result = []  # List[List[]] sentence without START and END
        result_prob = [] # List[prob] probs
        
        # infer one data at one time
        for batch_idx in range(batch_size):
            candidates = []
            
            # input image feature
            b_img_feature = img_embed_feature[batch_idx,:].reshape(1, -1)
            h0, c0 = self.lstm_cell(b_img_feature)
            
            # copy the initial candidates k times
            start = torch.LongTensor((1)).fill_(self.config.start_idx).to(self.config.device)
            for beam_idx in range(beam_size):
                candidates.append((1, [start.clone()], h0.clone(), c0.clone(), False)) #(prob, sentence, h, c, is_ended)
            
            # start decoding
            for t in range(self.config.max_sequence - 1):
                probs, sentences, h_input, c_input, is_ended = list(zip(*candidates))
                
                # early break if all candidates reached end
                if sum(is_ended) == len(candidates):
                    break

                x_input = torch.zeros((len(candidates), ), dtype=torch.long).to(self.config.device)
                for ii, sentence in enumerate(sentences):
                    x_input[ii] = sentence[-1]
                x_input = self.embedding(x_input)
                h_input = torch.cat(h_input, dim=0)
                c_input = torch.cat(c_input, dim=0)
                h, c = self.lstm_cell(x_input, (h_input, c_input))
                t_pred_word = self.output_linear(self.output_dropout(h))
                t_prob = self.softmax(t_pred_word)
                sorted, indices = torch.sort(t_prob, dim = 1, descending=True)

                new_candidates = []
    
                # generate beams for each original candidate
                for ii in range(len(candidates)):
                    if is_ended[ii]:
                        new_candidates.append(candidates[ii])
                    else:
                        # genearte k beams
                        for beam_idx in range(beam_size):
                            new_prob = probs[ii] * sorted[ii][beam_idx].item()
                            new_word = indices[ii][beam_idx]
                            new_sentence = sentences[ii] + [new_word]
                            new_ended = new_word.item() == self.config.end_idx
                            new_candidates.append((new_prob, new_sentence, h[ii].view(1, -1), c[ii].view(1, -1), new_ended))
                new_candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = new_candidates[:beam_size]
            
            best_sent = candidates[0][1][1:-1]
            best_sent = [word.item() for word in best_sent]
            result.append(best_sent)
            result_prob.append(candidates[0][0])
        return result, result_prob




    def generate_final_feature(self, entity_feature, neighbor_feature_list, img_feature, eid = None):
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
        atten_mask = neighbor_num_expand <= seq_range_expand #####changed from <= to <s
#         atten_mask = seq_range_expand >= neighbor_num_expand #####changed from <= to <s

        
                
        atten_mask = atten_mask.unsqueeze(1)

        # generate attended_neighbor feature
        attend_neighbor_feature, attention = self.atten_layer(key=neighbor_feature, value=neighbor_feature,
                                                   query=entity_feature.view(batch_size, 1, img_hidden),
                                                   attn_mask=atten_mask)    # [batch, 1, img_hidden]
        
#         attend_neighbor_feature, _ = self.atten_layer2(h=entity_feature, context=neighbor_feature, mask=atten_mask, output_tilde=False)

        # concat three types of feature
        attend_neighbor_feature = attend_neighbor_feature.view(batch_size, img_hidden)
        final_feature = torch.cat([entity_feature, attend_neighbor_feature, img_feature], dim=-1)   # [batch, 3 * img_hidden]
        
        #temp = []
        #for oo in range(len(eid)):
        #    if int(eid[oo]) == 285101:
        #        temp.append(final_feature[oo])
        #        print("found attn")
        #if len(temp) > 0:
        #    print(torch.sum(temp[0][:2048] != temp[1][:2048]))
        #    print(torch.sum(temp[0][2048:4096] != temp[1][2048:4096]))
        #    print(torch.sum(temp[0][4096:] != temp[1][4096:]))
        return final_feature
