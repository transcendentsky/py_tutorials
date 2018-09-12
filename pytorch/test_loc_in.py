#coding:utf-8
import torch
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='./loc_inf')


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.bone = nn.Sequential(nn.Conv2d(1,16,kernel_size=3,padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(16,32,kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(32, 64, kernel_size=3,padding=1),
                                  nn.Conv2d(64,128,kernel_size=3)
                                  )
        self.classifier = nn.Linear(9*128,2)

    def forward(self,x):
        o = self.bone(x)
        o = self.classifier(o.view(x.size(0),-1))
        return o

net = Net()
net.train()
print("-----------------------")

# net.cuda()
# criterion = nn.CrossEntropyLoss(reduce=True)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
sche = lr_scheduler.CosineAnnealingLR(opt, T_max=4001)

class LocDataLoader(object):
    def __init__(self, size=10, batch_size=128):
        self.size = 10
        self.batch_size = batch_size
        self.yield_num = 1000
        self.pad_width = 1
        self._locs = [[3,4],[2, 2], [3, 1], [1, 0], [3, 3],
                              [3, 1],
                              [4, 0],
                              [4, 1],
                              [0, 2],
                              [4, 3]]

    def get_item(self):
        loc = [0,0]
        input = np.ones((self.size,self.size))
        idx = np.random.randint(0,10)
        loc = self._locs[idx].copy()
        assert loc[0] >=0 and loc[0]<5 and loc[1] >=0 and loc[1] <5 , "[???] {}".format(loc)
        center = 128 * np.ones((5,5))
        for i in range(5):
            for j in range(5):
                input[i+loc[0],j+loc[1]] = center[i,j]
        loc[0] -= 2
        loc[1] -= 2
        return input, loc

    def forward(self):
        inputs = []
        locs = []
        for i in range(self.batch_size):
            input, loc = self.get_item()
            inputs.append(input)
            locs.append(loc)
        inputs = np.array(inputs)
        inputs = np.expand_dims(inputs, 1)
        locs = np.array(locs)
        return inputs, locs

    def yield_data(self):
        while 1:
            yield self.forward()

class TestLocDataLoader(object):
    def __init__(self, size=10, batch_size=32):
        self.size = 10
        self.batch_size = batch_size
        self.yield_num = 1000
        self.pad_width = 1

    def get_item(self):
        input = np.ones((self.size,self.size))
        loc = np.random.randint(0,5,(2,))
        center = 128 * np.ones((5,5))
        for i in range(5):
            for j in range(5):
                input[i+loc[0],j+loc[1]] = center[i,j]
        loc[0] -= 2
        loc[1] -= 2
        return input, loc

    def forward(self):
        inputs = []
        locs = []
        for i in range(self.batch_size):
            input, loc = self.get_item()
            inputs.append(input)
            locs.append(loc)
        inputs = np.array(inputs)
        inputs = np.expand_dims(inputs, 1)
        locs = np.array(locs)
        return inputs, locs

    def yield_data(self):
        while 1:
            yield self.forward()

loader = TestLocDataLoader(batch_size=128)
testloader = TestLocDataLoader()
# print("test", loader.forward())
# exit(0)
for i in range(1001):
    tloss = 0
    batch_data = next(loader.yield_data())
    (inputs, locs) = batch_data
    opt.zero_grad()
    inputs, locs = torch.FloatTensor(inputs), torch.FloatTensor(locs)
    # inputs, locs = Variable(inputs).cuda(), Variable(locs).cuda()
    inputs, locs = Variable(inputs), Variable(locs)
    o = net(inputs)
    # print(o.size(), locs.size())
    loss = F.smooth_l1_loss(o,locs)
    tloss += loss.data.item()
    loss.backward()
    opt.step()
    print("                                    locs: {}  loss : {}".format(locs.size(), loss.data[0]), end='\r')
    if i % 200 == 0:
        tloss = tloss / 200.0
        print("\n[EVAL] tloss: {}".format(tloss))
        writer.add_scalar("train loss", tloss, i/200)
        tloss = 0
        # eval
        test_loss = 0
        net.eval()
        for j in range(100):
            inputs, locs = next(testloader.yield_data())
            inputs, locs = torch.FloatTensor(inputs), torch.FloatTensor(locs)
            inputs, locs = Variable(inputs), Variable(locs)
            # inputs, locs = Variable(inputs).cuda(), Variable(locs).cuda()
            o = net(inputs)
            loss = F.smooth_l1_loss(o,locs)
            test_loss += loss
        print("[EVAL] eval loss: {}".format(test_loss/100.0))
        print("lr={}".format(opt.param_groups[0]['lr']))
        writer.add_scalar("eval loss", test_loss/100.0, i/200)
        # test
        # inputs, locs = next(testloader.yield_data())
        # inputs, locs = torch.FloatTensor(inputs), torch.FloatTensor(locs)
        # inputs, locs = Variable(inputs), Variable(locs)
        # o = net(inputs)
        # print("o : {} {}
        net.train()
        sche.step(i)

torch.save(net.state_dict(), 'test.pth')
