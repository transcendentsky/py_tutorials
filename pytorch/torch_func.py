#coding:utf-8
import torch
import numpy as np
import torch.nn as nn

def m1():
    c = np.zeros((128, 3, 32, 32))
    a = np.ones((128))
    b = np.ones((128, 3, 32, 32))

    wa = torch.from_numpy(a)
    wb = torch.from_numpy(b)
    wc = torch.from_numpy(c)

    # wc = torch.addcmul(wc, 1, wa, wb)
    # print(wc)

def m2():
    w = torch.from_numpy(np.arange(8 * 9).reshape((8, 3, 3)))
    print(w)
    # w = w.view(w.size(0), -1)
    w = torch.abs(w)
    print(w)
    # sum1 = torch.sum(w,(0,1))
    # print(sum1)
    # print(sum1.size())
    # print(w)
    w = w.sum(0)

    print(w)
    print(w.size())


def m3():
    # torch max ?????/
    # test torch max ???/
    w = torch.from_numpy(np.random.random((2,3,3,3)))
    # maxpool = nn.MaxPool2d(kernel_size=2,stride=1)
    # print(w)
    # mw = maxpool(w)
    # print(mw)

    w = w.view(2,3,-1)

    maxx = w.max(2)[0]  # minimum
    rd = torch.rand_like(w)   # return the rands  sized as w
    # rd = torch.rand(w.size(), dtype=torch.double)
    print("w: ", w)
    print(maxx)
    # w = (w - minn) ** 2
    w = maxx[:,:,None] - w
    meann = w.mean()
    w =  w / meann
    print("processed w: ", w)
    assert w.size() == rd.size() , "Not same? "
    print(rd)

    prob = 0.1
    w = w * prob
    print("w: ", w)

    final = torch.le(rd, w)
    print(final)
    final = final.view(2,3,3,3)
    print(final)

    # 实验证明， 和预想中一样， 每一层的max都是独立的， 并不是贯穿整条的
    # dropout ， batch中的每个图片采取不同的drop策略

def m4():
    # Set the inital value of weights ?????
    class BasicConv(nn.Module):

        def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True,
                     bn=True, bias=False):
            super(BasicConv, self).__init__()
            self.out_channels = out_planes
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
            self.relu = nn.ReLU(inplace=True) if relu else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x


def m5():
    a = torch.randn(10)
    b = torch.randn(10)
    c = a * b
    print(a, b, c)
    print(c.shape)

m3()
# m5()