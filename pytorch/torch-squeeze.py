import torch

a = torch.rand((2,2,2))
print(a)
print(a.size())

unsqe = a.unsqueeze(2)
print(unsqe, unsqe.size())