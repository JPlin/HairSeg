import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from component.modules import *


class DFN(nn.Module):
    def __init__(self,
                 in_channels=5,
                 out_channels=64,
                 add_fc=True,
                 self_attention=False,
                 attention_plus=False,
                 debug=False,
                 back_bone='resnet101'):
        super(DFN, self).__init__()
        self.add_fc = add_fc  # if flatten and fc the last stage
        self.self_attention = self_attention  # if add self attention
        self.attention_plus = attention_plus
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

        # for normal
        self.down_channel = ConvLayer(
            2048 // self.expand, 128, kernel_size=1,
            stride=1)  # choose 128 or 512
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # for fc
        if self.add_fc:
            self.down_channel = ConvLayer(
                2048 // self.expand, 128, kernel_size=1, stride=1)
            self.fc1 = ConvLayer(
                128 * 16 * 16, 1024 * 2, kernel_size=1, stride=1)
            self.fc2 = ConvLayer(
                1024 * 2, 512 * 8 * 8, kernel_size=1, stride=1)

        # for self_attention
        if self.self_attention:
            feature_size = 512
            dim_k = feature_size // 8
            self.down_channel_attention = ConvLayer(
                2048 // self.expand, feature_size, kernel_size=3, stride=2)
            if self.attention_plus:
                self.RM = Dual_Attn(feature_size, dim_k)
            else:
                self.RM = Spatial_Attn(feature_size, dim_k)

        self.stage_1 = StageBlock(1, self.expand)
        self.score_map_1 = ConvLayer(512, 2, kernel_size=1, stride=1)
        self.stage_2 = StageBlock(2, self.expand)
        self.score_map_2 = ConvLayer(512, 2, kernel_size=1, stride=1)
        self.stage_3 = StageBlock(3, self.expand)
        self.score_map_3 = ConvLayer(512, 2, kernel_size=1, stride=1)
        self.stage_4 = StageBlock(4, self.expand)
        self.score_map_4 = ConvLayer(512, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x_1 = self.res_1(x)
        x_2 = self.res_2(x_1)
        x_3 = self.res_3(x_2)
        x_4 = self.res_4(x_3)
        if self.debug:
            print("resnet ouput size ", x_4.size())

        if self.add_fc:
            # cut channel --> flatten --> fc --> reshape : [b , 512 , 8 ,8]
            x_gp = self.down_channel(x_4)
            x_flatten = x_gp.view(x_gp.size()[0], -1, 1, 1)
            x_flatten = self.fc1(x_flatten)
            x_flatten = self.fc2(x_flatten)
            if self.debug:
                print("flatten fc size ", x_flatten.size())
            x_gp = x_flatten.view(x_flatten.size()[0], 512, 8, 8)

        elif self.self_attention:
            x_fc = self.down_channel_attention(x_4)
            x_gp = self.RM(x_fc)
        else:
            x_gp = self.down_channel(x_4)
            x_gp = self.avg_pool(x_gp)
            # if flatten reshape to [b , c , h ,w ]
            f_size = x_4.size()[2]
            x_gp = x_gp.repeat(1, 1, f_size // 2, f_size // 2)

        x = self.stage_4(x_4, x_gp)
        score_4 = self.score_map_4(x)
        if self.debug:
            print("stage 4's size ", x.size())
        x = self.stage_3(x_3, x)
        score_3 = self.score_map_3(x)
        if self.debug:
            print("stage 3's size ", x.size())
        x = self.stage_2(x_2, x)
        score_2 = self.score_map_2(x)
        if self.debug:
            print("stage 2's size ", x.size())
        x = self.stage_1(x_1, x)
        score_1 = self.score_map_1(x)
        if self.debug:
            print("stage 1's size ", x.size())
        return [score_1, score_2, score_3, score_4]


class StageBlock(nn.Module):
    def __init__(self, stage=1, expand=1):
        super(StageBlock, self).__init__()
        assert stage in [1, 2, 3, 4]
        if stage == 1:
            in_channels = 256 // expand
        elif stage == 2:
            in_channels = 512 // expand
        elif stage == 3:
            in_channels = 1024 // expand
        elif stage == 4:
            in_channels = 2048 // expand

        self.RRB_1 = RRB(in_channels, 512)
        self.CAB = CAB(512)
        self.RRB_2 = RRB(512, 512)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1, x2):
        x1 = self.RRB_1(x1)
        #x2 = self.upsample(x2)
        x2 = F.upsample_bilinear(x2, x1.size()[2:])
        x1 = self.CAB(x1, x2)
        x1 = self.RRB_2(x1)
        return x1


class CAB(nn.Module):
    def __init__(self, in_channels):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_1 = ConvLayer(
            in_channels * 2, in_channels // 6, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.conv1_2 = ConvLayer(
            in_channels // 6, in_channels, kernel_size=1, stride=1)
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
        x = self.conv1_1(x)
        residul = x
        x = self.conv3_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        sum = residul + x
        x = self.relu(sum)
        return x
