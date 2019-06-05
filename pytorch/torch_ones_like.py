import torch
from torch.distributions import Bernoulli

# zz = torch.zeros((3,4,5))
# oo = torch.ones(zz.size())
# oo = torch.ones_like(zz)
# print(oo)
#
# # test scatter
bs = 3
hw = 4
c = 3
rads = torch.randint(0, hw * hw, (bs,)).long()
print(rads)
rads = torch.randint(0, hw * hw, (1,)).long().repeat(bs)
print(rads)
# print(rads)
# z = torch.zeros(bs, hw*hw).scatter_(1, rads, 1)
# print(z)
# print(z.reshape((bs,c,hw,hw)))

# repeat along axis
index = Bernoulli(0.4).sample((c,)).repeat(bs).reshape(bs,-1)
print(index)
# print(index)
index = torch.unsqueeze(index, 2)
print(index)
# ar = index.repeat(1,hw,1)
# print(ar)
br = index.repeat(1,1,hw)
print(br)