import numpy as np

import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from PIL import Image

nn_Module = nn.Module

class Encoder(nn.Module):
  def __init__(self, **kwargs):
    super(Encoder, self).__init__(**kwargs)

  def forward(self, X, *args):
    raise NotImplementedError

class DotProductAttention(nn.Module):
  def __init__(self, dropout, **kwargs):
    super(DotProductAttention, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
      d = queries.shape[-1]
      scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
      self.attention_weights = masked_softmax(scores, valid_lens)
      return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
  def __init__(self, key_size, query_size, value_size, num_hiddens,
               num_heads, dropout, bias=False, **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.attention = DotProductAttention(dropout)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
    self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
    self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
    self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

  def forward(self, queires, keys, values, valid_lens):
    # what is transpose_qkv() ???
    queires = transpose_qkv(self.W_q(queries), self.num_heads)
    keys = transpose_qkv(self.W_k(keys), self.num_heads)
    values =  transpose_qkv(self.W_v(values), self.num_heads)

    # To repeat `num_heads` times.
    if valid_lens is not None:
      valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

    # (batch_size*num_heads, num_queries, num_hiddens/num_heads)
    output = self.attention(queries, keys, values, valiad_lens)

    # (batch_size, num_queries, num_hiddens)
    output_concat = transpose_output(output, self.num_heads)
    
    return self.W_o(output_concat)

class EncoderBlock(nn.Module):

  def __init__(self, key_size, query_size, value_size, num_hiddens
               norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
               dropout, use_bais=False, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)
    self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                        num_heads, dropout, use_bais)
    self.addnorm1 = AddNorm(norm_shape, dropout)

    # what is PositionWiseFFN() ???
    self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)

    self.addnorm2 = AddNorm(norm_shape, dropout)

  def forward(self, X, valid_lens):
    Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
    return self.addnorm2(Y, self.ffn(Y))

