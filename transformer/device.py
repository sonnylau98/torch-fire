import torch

def try_gpu(i=0):

  if torch.cudo.device_count() >= i + 1:
    return torch.device(f'cuda:{i}')
  return torch.device('cpu')