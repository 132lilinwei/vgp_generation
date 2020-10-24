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
            self.embedding.from_pretrained(torch.FloatTensor(word_embed), freeze=False)

        self.lstm = nn.LSTM(config.embed_size, config.hidden_size, num_layers=config.lstm_num_layers,
                            batch_first=True, bidirectional=False)
        
        self.tanh = nn.Tanh()

        # convert img_feature to hidden
        self.relu = nn.ReLU()
        self.img_to_hidden = nn.Linear(config.image_hidden, config.hidden_size)

    def forward(self, img_feature, sentence, sen_seq_len):
        """
        :param img_feature: image's extracted feature, [batch_size, image_hidden]
        :param sentence: target sentence for teacher forcing, [batch_size, seq_len]
                         and has already converted to LongTensor
        :param sen_seq_len: record each sentences' len, [batch_size]
        :return: score: [batch], semantic match score of sentence and image
        """

        batch_size = img_feature.size(0)

        # here convert image_hidden to embed_size
        img_embed_feature = self.tanh(self.img_to_hidden(img_feature))

        # encode sentence
        sentence_embed = self.embedding(sentence)
        if len(sen_seq_len.shape) == 2:
            sen_seq_len = sen_seq_len.reshape(batch_size)
        pack_sent = pack_padded_sequence(sentence_embed, sen_seq_len, batch_first=True, enforce_sorted=False)
        pack_out, (h, c) = self.lstm(pack_sent)
        # we simply use the last hidden unit in last layer
#         h = h.view(self.config.lstm_num_layers, -1, batch_size, self.config.hidden_size).permute(0, 2, 1, 3)[-1]\
#             .reshape(batch_size, -1)
        h = self.tanh(h[-1])
#         print(h)

        # compute semantic match score
        # dot product image_feature with encoded feature and feed to sigmoid
        score = torch.sigmoid(torch.sum(img_embed_feature * h, dim=1))
        return score