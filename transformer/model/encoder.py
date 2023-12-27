import math
import pandas as pd
import torch
from torch import nn

def masked_softmax(X, valid_lens):
  def _sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
  
  if valid_lens is None:
    return nn.functional.softmax(X, dim=-1)
  else:
    shape = X.shape
    if valid_lens.dim() == 1:
      valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
      valid_lens = valid_lens.reshape(-1)
    X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
  def __init__(self, dropout):
    super().__init__()
    self.dropout = nn.Dropout(dropout)

  def forward(self, queries, keys, values, valid_lens=None):
    d = queries.shape[-1]
    scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
    self.attention_weights = masked_softmax(scores, valid_lens)
    return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
  def __init__(self, num_hiddens,
               num_heads, dropout, bias=False, **kwargs):
    super().__init__()
    self.num_heads = num_heads
    self.attention = DotProductAttention(dropout)
    self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
    self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
    self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
    self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

  def forward(self, queries, keys, values, valid_lens):
    queries = self.transpose_qkv(self.W_q(queries))
    keys = self.transpose_qkv(self.W_k(keys))
    values = self.transpose_qkv(self.W_v(values))
    if valid_lens is not None:
      valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
    output = self.attention(queries, keys, values, valid_lens)
    output_concat = self.transpose_output(output)
    return self.W_o(output_concat)
  
  def transpose_qkv(self, X):
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

  def transpose_output(self, X):
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionWiseFFN(nn.Module):
  def __init__(self, ffn_num_hiddens, ffn_num_outputs):
    super().__init__()
    self.dense1 = nn.LazyLinear(ffn_num_hiddens)
    self.relu = nn.ReLU()
    self.dense2 = nn.LazyLinear(ffn_num_outputs)

  def forward(self, X):
    return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
  def __init__(self, norm_shape, dropout):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.ln = nn.LayerNorm(norm_shape)

  def forward(self, X, Y):
    return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
  def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
               use_bias=False):
    super().__init__()
    self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                        dropout, use_bias)
    self.addnorm1 = AddNorm(num_hiddens, dropout)
    self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
    self.addnorm2 = AddNorm(num_hiddens, dropout)

  def forward(self, X, valid_lens):
    Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
    return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(Encoder):
  def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
               num_heads, num_blks, dropout, use_bias=False):
    super().__init__()
    self.num_hiddens = num_hiddens
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_blks):
      self.blks.add_module("block" + str(i), TransformerEncoderBlock(
        num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias
      ))    
    self.dense = nn.LazyLinear(vocab_size)

  def forward(self, X, valid_lens):
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    self.attention_weights = [None] * len(self.blks)
    for i, blk in enumerate(self.blks):
      X = blk(X, valid_lens)
      self.attention_weights[i] = blks.attention.attention.attention_weights
    return X