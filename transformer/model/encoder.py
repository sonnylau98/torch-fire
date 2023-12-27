class TransformerEncoderBlock(nn.Module):

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