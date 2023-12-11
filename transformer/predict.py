def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):

  net.eval()
  src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
  enc_valid_len = # truncate_pad

  enc_X = torch.unsqueeze(
    torch.tensor(src_tokens, dtype=torch.long, device=device, dim=0)
  )
  enc_outputs = net.encoder(enc_X, enc_valid_len)
  dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

  dec_X = torch.unsqueeze(
    torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device, dim=0)
  )
  output_seq, attention_weight_seq = [], []
  for _ in range(num_steps):
    Y, dec_state = net.decoder(dec_X, dec_state)

    dec_X = Y.argmax(dim=2)
    pred = dec_X.squeeze(dim=0).type(torch.int32).item()

    if save_attention_weights:
      attention_weight_seq.append(net.decoder.attention_weights)
      
    if pred == tgt_vocab['<eos>']:
      break

    output_seq.append(pred)
  
  return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
  