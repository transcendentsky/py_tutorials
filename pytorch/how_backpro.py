#coding:utf-8
import torch
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

class CustomDataLoader(data.Dataset):
    def __init__(self):
        self.nums = np.arange(100)
        self.xs = np.arange(100)
        self.ys = np.arange(100)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def __len__(self):
        return len(self.xs)

transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
trainset = torchvision.datasets.CIFAR10(root='/media/trans/mnt/code_test/my_projects/mixup-master/cifar/data',train=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=8, shuffle=False)

from torch.optim.optimizer import Optimizer, required
class SGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                print(d_p)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bone = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(3, 16, kernel_size=3),
                                  nn.MaxPool2d(kernel_size=2,stride=2),
                                  nn.Conv2d(16,32,kernel_size=3)) # 32,> 16,> 14 > 7 > 5
        self.classifier = nn.Linear(5*5*32,10)

    def forward(self, x):
        out0 = self.bone(x)
        out1 = out0.view(out0.size(0), -1)
        out = self.classifier(out1)
        return out, out0, out1


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.c1 = nn.Linear(4,2)
        self.c2 = nn.Linear(2,2)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        out0 = self.c1(x)
        out1 = self.c2(out0)
        return out1, out0 # 最后输出sparse 参数



net = Net()
net.train()
print("-----------------------")
# for name, param in net.state_dict().items():
#     print(name)
#     net.state_dict()[name].copy_(torch.ones_like(net.state_dict()[name]))
# net.state_dict()['c1.weight'].copy_(torch.FloatTensor(np.array([[1.1,2.1,3,3],[2,0,3,0.5]])))
# net.state_dict()['c2.weight'].copy_(torch.FloatTensor(np.array([[0.1,2.1],[2,1]])))

# print(net.state_dict())

# net.cuda()
criterion = nn.CrossEntropyLoss(reduce=True)
opt = SGD(net.parameters(), lr=0.01)

inputs = np.array([[0,1,2,3],[3,3,4,3]]).reshape(2,4)
targets = np.array([0,0])
# inputs = np.array([0,1,2,3]).reshape((1,4))
# targets = np.array([0]).reshape((1,))

# inputs = np.array([3,3,4,3]).reshape((1,4))
# targets = np.array([0]).reshape((1,))
'''
inputs, targets = torch.FloatTensor(inputs), torch.LongTensor(targets)
inputs, targets = Variable(inputs), Variable(targets)
opt.zero_grad()
outputs, out0 = net(inputs)
print("ooooooooooooooooooooooo")
print(out0)
print(outputs)

loss = criterion(outputs, targets)
print(loss)
# print(loss)
print("-----------------------")

print("-----------------------")
loss = loss.mean()
loss.backward()
opt.step()
print("end")
'''

# RuntimeError: grad can be implicitly created only for scalar outputs
# Loss.backward() loss can only be a single tensor
for i in range(1000000):
    for idx, (inputs, targets) in enumerate(trainloader):
        # inputs, targets = images.cuda(), labels.cuda()
        inputs, targets = Variable(inputs), Variable(targets) # FloatTensor, LongTensor
        # print(inputs.size())
        # print(targets.size())
        opt.zero_grad()
        outputs, out0, out1 = net(inputs)
        loss = criterion(outputs, targets)
        try:
            loss2 = loss.mean()
            raise Exception
        except:
            # print "-" ,
            continue
        loss.backward()
        # print("*********************")
        # print(outputs.grad_fn)
        # print(out0)

        opt.step()
    print("Training ********")
