class TransformerDecoderBlock(nn.Module):
  def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
    super().__init__()
    self.i = i
    self.attention1 = MultiHeadAttention(num_hiddens, num_heads,
                                         dropout)
    self.addnorm1 = AddNorm(num_hiddens, dropout)
    self.attention2 = MultiHeadAttention(num_hiddens, num_heads,
                                         dropout)
    self.addnorm2 = AddNorm(num_hiddens, dropout)
    self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
    self.addnorm3 = AddNorm(num_hiddens, dropout)

  def forward(self, X, state):
    enc_outputs, enc_valid_lens = state[0], state[1]
    if state[2][self.i] is None:
      key_values = X
    else:
      key_values = torch.cat((state[2][self.i], X), dim=1)
    state[2][self.i] = key_values
    if self.training:
      batch_size, num_steps, _ = X.shape
      dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
    else:
      dec_valid_lens = None
    X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
    Y = self.addnorm1(X, X2)
    Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
    Z = self.addnorm2(Y, Y2)
    return self.addnorm3(Z, self.ffn(Z)), state