import torch
import numpy as np

c = np.zeros((128,3,32,32))
a = np.ones((128))
b = np.ones((128,3,32,32))

wa = torch.from_numpy(a)
wb = torch.from_numpy(b)
wc = torch.from_numpy(c)

wc = torch.addcmul(wc, 1, wa, wb)
print wc