'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable


vggcfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, in_channel=3, out_channel=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(vggcfg[vgg_name], in_channel=in_channel)
        self.classifier = nn.Linear(512, out_channel)
        # self.extra_classifier = False
        # if out_channel != 10:
        #     self.extra_classifier = True
        #     self.classifier2 = nn.Linear(512, out_channel)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out1 = self.classifier(out)
        # if self.extra_classifier:
        #     out2 = self.classifier2(out)
        #     return out2
        return out1

    def _make_layers(self, cfg, in_channel=3):
        layers = []
        in_channels = in_channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


net = VGG('VGG16')
net.cuda()
net = torch.nn.DataParallel(net)

def init_net():
    #######  Parameters Initialization  #########
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            print("####  Using xavier Initializer  ####")
            nn.init.xavier_normal(m.weight.data)
    if True:
        net.apply(weights_init)
    print("[***] Initialized Net")

init_net()
print(net)

print("Saving")
torch.save(net.state_dict(), "test.pth")
print("Loading")
checkpoint = torch.load("test.pth")

print(checkpoint.keys())