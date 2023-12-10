import numpy as np

import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from PIL import Image

################################################
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
################################################


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

    queires = transpose_qkv(self.W_q(queires), self.num_heads)
    keys = transpose_qkv(self.W_k(keys), self.num_heads)
    values =  transpose_qkv(self.W_v(values), self.num_heads)

    # To repeat `num_heads` times.
    if valid_lens is not None:

      #Evalid_lensRROR ???
      valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

    # (batch_size*num_heads, num_queries, num_hiddens/num_heads)
    output = self.attention(queries, keys, values, valiad_lens)

    # (batch_size, num_queries, num_hiddens)
    output_concat = transpose_output(output, self.num_heads)
    
    return self.W_o(output_concat)

################################################
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状

    Defined in :numref:`sec_multihead-attention`"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])
################################################


class EncoderBlock(nn.Module):

  def __init__(self, key_size, query_size, value_size, num_hiddens,
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


class PositionalEncoding(nn.Module):
  def __init__(self, num_hiddens, dropout, max_len=1000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(dropout)

    self.P = torch.zeros((1, max_len, num_hiddens))
    
    X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

    self.P[:, :, 0::2] = torch.sin(X)
    self.P[:, :, 1::2] = torch.cos(X)


  def forward(self, X):
    X = X + self.P[:, :X.shape[1], :].to(X.device)
    return self.dropout(X)


class TransformerEncoder(Encoder):

  def __init__(self, vocab_size, key_size, query_size, value_size,
               num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
               num_heads, num_layers, dropout, use_bias=False, **kwargs):
    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_hiddens = num_hiddens
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_layers):
      self.blks.add_module("block"+str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                                        norm_shape, ffn_num_input, ffn_num_hiddens,
                                                        num_heads, dropout, use_bias))

  def forward(self, X, valid_lens, *args):
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    self.attention_weights = [None] * len(self.blks)
    for i, blk in enumerate(self.blks):
      X = blk(X, valid_lens)
      self.attention_weights[i] = blk.attention.attention.attention_weights
    return X


class DecoderBlock(nn.Module):
  def __init__(self, key_size, query_size, value_size, num_hiddens,
               norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
               dropout, i, **kwargs):
    super(DecoderBlock, self).__init__(**kwargs)
    self.i = i
    self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
    self.addnorm1 = AddNorm(norm_shape, dropout)
    self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
    self.addnorm2 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
    self.addnorm3 = AddNorm(norm_shape, dropout)

  # state ???
  def forward(self, X, state):
    enc_outputs, enc_valid_lens = state[0],state[1]

    if state[2][self.i] is None:
      key_values = X
    else: 
      key_values = torch.cat((state[2][self.i], X), axis=1)
    
    state[2][self.i] = key_values

    if self.training:
      batch_size, num_steps, _ = X.norm_shape

      dec_valid_lens = torch.arange(1, num_steps + 1. device=X.device).repeat(batch_size, 1)
    else:
      dec_valid_lens = None

    X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
    Y = self.addnorm1(X, X2)
    Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
    Z = self.addnorm2(Y, Y2)
    return self.addnorm3(Z, self.ffn(Z)), state

    
#########################################

class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口

    Defined in :numref:`sec_seq2seq_attention`"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

#########################################

class TranformerDecoder(AttentionDecoder):
  def __init__(self, vocab_size, key_size, query_size, value_size,
               num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
               num_heads, num_layers, dropout, **kwargs):
    super(TranformerDecoder, self).__init__(**kwargs)
    self.num_hiddens = num_hiddens
    self.num_layers = num_layers
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_layers):
      self.blks.add_module("block"+str(i), DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                                        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                                                        dropout, i))
    self.dense = nn.Linear(num_hiddens, vocab_size)

  def init_state(self, enc_outputs, enc_valid_lens, *args):
    return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

  def forward(self, X, state):
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
    for i, blk in enumerate(self.blks):
      X, state = blk(X, state)
      self._attention_weights[0][i] = blk.attention1.attention.attention_weights
      self._attention_weights[1][i] = blk.attention2.attention.attention_weights
    return self.dense(X), state

  @property
  def attention_weights(self):
    return self._attention_weights


