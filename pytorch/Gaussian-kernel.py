import numpy as np
import torch

x, y = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
d = np.sqrt(x*x+y*y)
sigma, mu = 1, 0.0
g = np.clip(np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * 1.25 , 0.0, 1.0)
g = g[np.newaxis, np.newaxis, :, :]
print("2D Gaussian-like array:")
print(g)

tg = torch.from_numpy(g)
print(tg)
print(tg.size())