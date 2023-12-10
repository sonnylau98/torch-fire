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