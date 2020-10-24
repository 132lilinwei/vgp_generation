import torch
import torch.nn as nn

class generator(nn.Module):
    def __init__(self, config, word_embed=None):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_idx)
        if word_embed is not None:
            self.embedding.from_pretrained(torch.FloatTensor(word_embed), freeze=False)

        self.lstm_cell = nn.LSTMCell(config.embed_size, config.hidden_size, bias=True)
        
        # used for multinomial sampling
        self.softmax = nn.Softmax(dim=1)

        # convert img_feature to hidden
        self.relu = nn.ReLU()
        self.img_to_embed = nn.Linear(config.image_hidden, config.embed_size)

        # convert lstm hidden to predicted vocab
        self.output_dropout = nn.Dropout(config.dropout)
        self.output_linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, img_feature, target_sent=None, target_sent_len=None, fix_seq_len=-1):
        """
        Here we do not add noise
        :param img_feature: image's extracted feature, [batch_size, image_hidden]
        :param target_sent: target sentence for teacher forcing, [batch_size, seq_len]
                            assume [START, sentence, END], and has already converted to LongTensor
        :param target_seq_len: record each sentences' len, [batch_size]
        :param fix_seq_len: if t > fix_seq_len, use multinomial sampling rather than groudtruth
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
        img_embed_feature = self.relu(self.img_to_embed(img_feature))
        h, c = self.lstm_cell(img_embed_feature)

        # then in the training phase, we use target_sent for teacher forcing
        if target_sent is not None:
            # here is training phase, where provide target sentence

            # use sampled_sent because we will change the words
            sampled_sent = target_sent.clone()
            seq_len = target_sent.size(1)
            if fix_seq_len == -1:
                fix_seq_len = seq_len
            sampled_sent_embed = self.embedding(sampled_sent) # [batch_size, seq_len, embed_size]

            # iterate whole sequence
            prev_sampled = None
            for t in range(seq_len - 1):    # seq_len - 1 means ignore END
                t_sent_embed = sampled_sent_embed[:, t, :] if prev_sampled is None else self.embedding(prev_sampled)
                t_sent_embed = t_sent_embed.reshape(batch_size, -1)
                h, c = self.lstm_cell(t_sent_embed, (h, c))
                t_pred_word = self.output_linear(self.output_dropout(h)) # [batch_size, vocab]
                pred_words_probs.append(t_pred_word)
                if t >= fix_seq_len:
                    # if t step is larger than fix_seq_len, we should use multinomial sample words
                    # will use target_sent_len to mask padding case.
                    prev_sampled = torch.multinomial(self.softmax(t_pred_word), 1)   # [batch_size, 1]
                    sampled_sent[:, t + 1] = prev_sampled.reshape(batch_size).clone()
                else:
                    prev_sampled = None
        else:
            # here is inference time to generate caption
            # first add START in the begining
            sampled_sent = [torch.LongTensor((batch_size)).fill_(self.config.start_idx).to(self.config.device)]
            for t in range(self.config.max_sequence - 1):
                input_seq = sampled_sent[-1]

                input_seq_embed = self.embedding(input_seq)
                h, c = self.lstm_cell(input_seq_embed, (h, c))
                t_pred_word = self.output_linear(self.output_dropout(h))
                pred_words_probs.append(t_pred_word)

                # use multinomial sampling to choose
                sampled = torch.multinomial(self.softmax(t_pred_word), 1) # [batch_size, 1]
                sampled_sent.append(sampled.reshape(batch_size))
            sampled_sent = torch.stack(sampled_sent).permute(1, 0).to(self.config.device)  # [batch_size, seq_len]

        # pred_words_probs: [batch_size, seq_len, vocab_size]
        pred_words_probs = torch.stack(pred_words_probs).permute(1, 0, 2).to(self.config.device)

        # means sampled_sent's len for each example
        # For each example, find the idx of end_idx and set len=idx+1. if miss, set len=seq_len
        sampled_sent_len = torch.stack([(sent==self.config.end_idx).nonzero()[0] + 1
                                        if len((sent==self.config.end_idx).nonzero()) > 0
                                        else torch.LongTensor([sampled_sent.size(1)]).to(self.config.device)
                                        for sent in sampled_sent])
        sampled_sent_len = sampled_sent_len.reshape(-1).to(self.config.device)

        return pred_words_probs, sampled_sent, sampled_sent_len

    def greedy_generate(self, img_feature):
        """
        We use greedy policy to generate the sentence
        Each timestep t we choose the most possible token
        :param img_feature: image's extracted feature, [batch_size, image_hidden]
        :return: final_sent: List[List[]], each element (List) means i-th generated sentence, without END and START
        """
        batch_size = img_feature.size(0)
        pred_sents = [torch.LongTensor((batch_size)).fill_(self.config.start_idx).to(self.config.device)]

        # in the first timestep, we feed in the image feature
        # here convert image_hidden to embed_size
        img_embed_feature = self.relu(self.img_to_embed(img_feature))
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