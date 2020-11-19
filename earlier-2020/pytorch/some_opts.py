#coding:utf-8

import torch

"""
torch.unsqueeze(input, dim, out=None) → Tensor
Returns a new tensor with a dimension of size one inserted at the specified position.

The returned tensor shares the same underlying data with this tensor.
"""

x = torch.Tensor([1, 2, 3, 4])
xx = torch.Tensor([[2,2,2,2],[2,2,2,2]])
x1 = torch.unsqueeze(x, 0) # insert dim 0
x2 = torch.unsqueeze(x, 1) # insert dim 1
print(x1)
print(x2)

a = torch.Tensor([1,2,2,3])
b = torch.Tensor([2,2,2,2])
amax = torch.max(a,b)
print(amax)
print(torch.max(a))

xe = x.unsqueeze(0)
xe = xe.expand_as(xx)
print(xe)