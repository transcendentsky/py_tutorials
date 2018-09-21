import torch

num = 32
num_priors = 8732
print(num, num_priors)
loc_t = torch.Tensor(num, num_priors, 4)
print(loc_t)


t = torch.cat(1,1)
print(t)