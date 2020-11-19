import torch
import torch.nn as nn
import numpy as np

# multi-head attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        model_dim = config.image_hidden    # original hidden dimension
        num_heads = config.num_heads

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(config)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(config.dropout)

        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        """
        :param key: [Batch, box_num, hidden], all b-box in the image
        :param value: [Batch, box_num, hidden]
        :param query: [Batch, 1, hidden], target b-box
        :param attn_mask: [batch, 1, box_num]
        :return:
            output: [batch, 1, hidden]
        """
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.reshape(batch_size, -1, num_heads, dim_per_head).permute(0, 2, 1, 3)\
            .reshape(batch_size*num_heads, -1, dim_per_head)
        value = value.reshape(batch_size, -1, num_heads, dim_per_head).permute(0, 2, 1, 3)\
            .reshape(batch_size*num_heads, -1, dim_per_head)
        query = query.reshape(batch_size, -1, num_heads, dim_per_head).permute(0, 2, 1, 3)\
            .reshape(batch_size*num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1).view(batch_size*num_heads, 1, -1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.reshape(batch_size, num_heads, -1, dim_per_head).permute(0, 2, 1, 3)\
            .reshape(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

# Only attention layer
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config

        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        q: Queries张量，形状为[B, L_q, D_q]
        k: Keys张量，形状为[B, L_k, D_k]
        v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        attn_mask: Masking张量，形状为[B, L_q, L_k], 0-1 values

        Returns
        context: [B, L_q, D_k]
        attention: [B, L_q, L_k]
        """
        # (QK^T) / sqrt(d_k)
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale

        # set attention mask, set to -inf
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)

        # compute softmax
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # attention * V
        context = torch.bmm(attention, v)
        return context, attention

    
class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, config):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(config.image_hidden, config.image_hidden, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(config.image_hidden + config.image_hidden, config.image_hidden, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn
