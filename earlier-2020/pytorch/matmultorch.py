import torch

a=torch.randint(11,(3,2,2))
b=torch.randint(11,(3,2,2))
print(a,b)
# print(torch.mm(a,b))
print(torch.matmul(a,b))
print(a*b)

# batch1 = torch.randn(2,3,4)
# batch2 = torch.randn(2,4,5)
batch1=torch.randint(10,(3,2,2))
batch2=torch.randint(3,(3,2,2))
res = torch.bmm(batch1, batch2)
print(batch1, batch2)
print(res.size(), res)