import torch
import numpy as np
import torchvision
import torch.nn as nn   
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F


class IPT(nn.Module):
    def __init__(self, backbone, transformer, image_decoder):
        super().__init__() 
        self.backbone = backbone
        self.transformer = transformer
        self.f2image = image_decoder
        
    def forward(self, x ):
        # (3,128,128) => (3*4 , 128, 128)
        f = self.backbone(x)  
        import ipdb; ipdb.set_trace()
        # (b c m n) => (b c2 m n)
        f_patch = self.get_patch(f)
        f2_patch = self.transformer(f_patch)
        f2 = self.combine(f2_patch)
        image = f2image(f2)
        import ipdb; ipdb.set_trace()
        pass
        
        return x
        
    def get_patch(self, x ):
        # TODO
        return x
        pass
    
    def combine(self, x):
        # TODO
        return x
        pass 
        

class SimpleImageDecoder(nn.Module):
    def __init__(self, input_dim=256, sr_size=(96,96), times=2):
        super().__init__()
        self.output_dim = 3 * times
        self.sr_size = sr_size
        self.kernel_size = 3
        # self.linear_1 = nn.Linear(input_dim, output_dim//2, bias=True)
        # self.linear_2 = nn.Linear(output_dim//2, output_dim, bias=True)
        self.linear_m1 = nn.Linear(100, 1000, bias=True)
        self.linear_m2 = nn.Linear(1000, 96*96, bias=True)
        self.linear_c1 = nn.Linear(256, 3*times, bias=True)
        
        # self.conv1 = nn.Conv2d(256,          output_dim*2, self.kernel_size, padding=1, stride=1)
        # self.conv2 = nn.Conv2d(output_dim*2, output_dim*2, self.kernel_size, padding=1, stride=1)
        # self.conv3 = nn.Conv2d(output_dim*2, output_dim,   self.kernel_size, padding=1, stride=1)
        # self.gn1   = torch.nn.GroupNorm(8, output_dim*2)
        # self.gn2   = torch.nn.GroupNorm(8, output_dim*2)
        
    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.linear_m1(x)
        x = self.linear_m2(x)
        x = x.transpose(-1, -2)
        x = self.linear_c1(x)
        x = x.transpose(-1, -2)
        b, c, mn = x.size()
        # import ipdb; ipdb.set_trace()
        assert mn == self.sr_size[0]*self.sr_size[1], f"mn: {mn}, sr: {self.sr_size}"
        x = x.view(b, self.output_dim, self.sr_size[0], self.sr_size[1])
        return x

