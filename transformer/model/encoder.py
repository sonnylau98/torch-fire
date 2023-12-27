from torch.nn import MultiheadAttention

################################################
class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """The residual connection followed by layer normalization."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
################################################

class TransformerEncoderBlock(nn.Module):
  def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
               use_bias=False):
    super().__init__()
    self.attention = MultiheadAttention(num_hiddens, num_heads,
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