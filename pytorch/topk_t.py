import torch

x = torch.arange(1., 6.)
print(x)
top3 = torch.topk(x, 3)
print(top3)
