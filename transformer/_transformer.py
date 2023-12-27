import numpy as np

import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from PIL import Image







class DotProductAttention(nn.Module):
  def __init__(self, dropout, **kwargs):
    super(DotProductAttention, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)

  def forward(self, queries, keys, values, valid_lens=None):
    d = queries.shape[-1]
    scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
    self.attention_weights = masked_softmax(scores, valid_lens)
    return torch.bmm(self.dropout(self.attention_weights), values)


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


##########################################
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
##########################################

