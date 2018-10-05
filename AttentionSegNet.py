import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from component.modules import *


class AttnSegNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 debug=False,
                 back_bone='resnet101'):
        super(AttnSegNet, self).__init__()
        self.debug = debug
        self.conv1 = ConvLayer(
            in_channels, out_channels, kernel_size=3, stride=2)


class SegBranch(nn.Module):
    def __init__(self, in_channels, out_channels, debug, back_bone):
        super(SegBranch, self).__init__()
        self.debug = debug
        self.conv1 = ConvLayer(
            in_channels, out_channels, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if back_bone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            self.expand = 1
        elif back_bone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.expand = 1
        elif back_bone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            self.expand = 4
        elif back_bone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.expand = 4
        else:
            raise "undefined backbone"

        self.res_1 = resnet.layer1
        self.res_2 = resnet.layer2
        self.res_3 = resnet.layer3
        self.res_4 = resnet.layer4

    def forward(self, x):
        return None

    def warp_distort(self, x, fa_points, std_points, out_size=(512, 512)):
        ratio_y, ratio_x = gen_
        return None

    def un_warp_distort(self, x):
        return None