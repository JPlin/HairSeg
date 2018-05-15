import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class DFN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        self.conv1 = ConvLayer(
            in_channels, out_channels, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resnet101 = models.resnet101(pretrained=True)
        self.res_1 = resnet101.layer1
        self.res_2 = resnet101.layer2
        self.res_3 = resnet101.layer3
        self.res_4 = resnet101.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.stage_1 = StageBlock(1)
        self.stage_2 = StageBlock(2)
        self.stage_3 = StageBlock(3)
        self.stage_4 = StageBlock(4)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv2 = ConvLayer(512, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x_1 = self.res_1(x)
        x_2 = self.res_2(x_1)
        x_3 = self.res_3(x_2)
        x_4 = self.res_4(x_3)
        x_gp = self.avg_pool(x_4)
        x = self.stage_4(x_4, x_gp)
        x = self.stage_3(x_3, x)
        x = self.stage_2(x_2, x)
        x = self.stage_1(x_1, x)
        x = self.upsample(x)
        x = self.conv2(x)
        return x


class StageBlock(nn.Module):
    def __init__(self, stage=1):
        assert stage in [1, 2, 3, 4]
        if stage == 1:
            in_channels = 64
        elif stage == 2:
            in_channels = 128
        elif stage == 3:
            in_channels = 256
        elif stage == 4:
            in_channels = 512

        self.RRB_1 = RRB(in_channels, 512)
        self.CAB = CAB(512)
        self.RRB_2 = RRB(512, 512)
        if stage == 4:
            self.upsample = None
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1, x2):
        x1 = self.RRB_1(x1)
        if self.upsample:
            x2 = self.upsample(x1)
        else:
            f_size = x1.size()[2]
            x2 = x2.repeat(1, 1, f_size, f_size)
        x1 = self.CAB(x1, x2)
        x1 = self.RRB_2(x1)
        return x1


class CAB(nn.Module):
    def __init__(self, in_channels):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_1 = ConvLayer(
            in_channels, in_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.conv1_2 = ConvLayer(
            in_channels, in_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x_ = torch.cat([x1, x2], dim=1)
        # global average pool
        x_ = self.avg_pool(x_)
        x_ = self.conv1_1(x_)
        x_ = self.relu(x_)
        x_ = self.conv1_2(x_)
        x_ = self.sigmoid(x_)  # output N * C ?
        # x_ = torch.unsqueeze(x_)  # output N * C * 1
        # x_ = torch.unsqueeze(x_)  # output N * C * 1 * 1
        x_ = x_ * x1
        x_ = x_ + x2
        return x_


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1_1 = ConvLayer(
            in_channels, out_channels, kernel_size=1, stride=1)
        self.conv3_1 = ConvLayer(
            out_channels, out_channels, kernel_size=3, stride=1)
        self.conv3_2 = ConvLayer(
            out_channels, out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        residul = x
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        sum = residul + x
        x = self.relu(sum)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out