import torch
import numpy as np

x = torch.FloatTensor(np.zeros((3,32,32)))


a = torch.zeros_like(x)


print(x.size())
print(x)