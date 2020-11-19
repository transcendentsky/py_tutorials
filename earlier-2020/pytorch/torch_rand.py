import torch
import numpy as np

npx = np.array([[1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

npx2 = 1 - npx

npt = torch.from_numpy(npx[np.newaxis, np.newaxis, :, :]).float()
rr = torch.randn_like(npt)*3
print(rr)
