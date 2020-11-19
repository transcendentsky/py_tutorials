import torch

"""
torch.eq()   in == out
torch.ne()   in =! out

torch.gt()   in > out
torch.ge()   in >= out

torch.le()   in <= out
torch.lt()   in < out

"""

a = torch.Tensor([[1, 2, 3], [3, 2, 4]])
b = torch.Tensor([[2, 2, 2], [4, 4, 4]])

t = torch.eq(a,b)
count = t.sum()
print(t)
print(count.item())