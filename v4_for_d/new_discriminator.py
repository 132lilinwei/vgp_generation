import torch
import torch.nn as nn
from torch.nn.utils.rnn import *

class discriminator(nn.Module):
    def __init__(self, config, word_embed=None):
        super().__init__()

        self.config = config

        # Encode sentence
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_idx)
        if word_embed is not None:
            self.embedding.from_pretrained(torch.FloatTensor(word_embed), freeze=False, padding_idx=config.pad_idx)

        self.lstm = nn.LSTM(config.embed_size, config.d_hidden_size, num_layers=config.lstm_num_layers,
                            batch_first=True, bidirectional=False)

        # convert img_feature to hidden
        self.img_to_embed = nn.Linear(config.image_hidden, config.d_hidden_size)

        self.linear1 = nn.Linear(2 * config.d_hidden_size, config.d_hidden_size)
        self.linear2 = nn.Linear(config.d_hidden_size, 3)

    def forward(self, img_feature, sentence, sen_seq_len):
        """
        :param img_feature: image's extracted feature, [batch_size, image_hidden]
        :param sentence: target sentence for teacher forcing, [batch_size, seq_len]
                         and has already converted to LongTensor
        :param sen_seq_len: record each sentences' len, [batch_size]
        :return: out: [batch, 3], 3 means [target, generated, random_sampled]
        """

        batch_size = img_feature.size(0)

        # here convert image_hidden to embed_size
        img_embed_feature = torch.tanh(self.img_to_embed(img_feature))

        # encode sentence
        sentence_embed = self.embedding(sentence)
        if len(sen_seq_len.shape) == 2:
            sen_seq_len = sen_seq_len.reshape(batch_size)
        pack_sent = pack_padded_sequence(sentence_embed, sen_seq_len, batch_first=True, enforce_sorted=False)
        
        pack_out, (h, c) = self.lstm(pack_sent)
        # we simply use the last hidden unit in last layer
        h = torch.tanh(h.squeeze(0))

        # concat
        concat = torch.cat((img_embed_feature, h), dim=-1)  # [batch, 2 * hidden]
        out = self.linear2(torch.tanh(self.linear1(concat)))
        return out

