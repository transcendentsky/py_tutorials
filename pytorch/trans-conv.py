import numpy as np
import torch
import torch.nn.functional as F


npx = np.array([[1, 1, 0, 0],
          [1, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]])

npx2 = 1 - npx

npt = torch.from_numpy(npx[np.newaxis,np.newaxis,:,:]).float()
npt2 = torch.from_numpy(npx2[np.newaxis,np.newaxis,:,:]).float()

expandsize = 7
deconv = F.conv2d(npt, torch.ones((1, 1, expandsize, expandsize)), padding=6)
deconv2 = F.conv2d(npt2, torch.ones((1, 1, expandsize, expandsize)), padding=6)

print(deconv)
print(deconv2)

dc12 = deconv2.numpy().squeeze()
dc12 = np.sqrt(dc12)
dc12 = dc12/np.max(dc12)
print(dc12)
dc1 = deconv.numpy().squeeze()
dc1 = np.sqrt(dc1)
print(dc1)

sum1 = deconv.sum()
sum2 = deconv2.sum()

print(sum1.item(), sum2.item())

partion = 6.0 / 25.0
partion2 = sum1/(sum1+sum2)

print(partion, partion2)

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(20180316)
x = np.random.randn(4,4)
f, (ax1) = plt.subplots(figsize=(6,6), nrows=1)
sns.despine(left=True, bottom=True, top=True, right=True, offset=None, trim=False)
sns.heatmap(dc12, annot=True, ax=ax1)

plt.show()