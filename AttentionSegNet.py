import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from component.modules import *
import matplotlib.pyplot as plt


class AttnSegNet(nn.Module):
    def __init__(self,
                 classes=2,
                 in_channels=3,
                 out_channels=64,
                 debug=False,
                 back_bone='resnet101'):
        super(AttnSegNet, self).__init__()
        self.debug = debug
        self.seg = SegBranch(in_channels, out_channels, debug, back_bone)
        self.attn = AttnBranch(in_channels, out_channels, debug, back_bone)
        self.conv1 = ConvLayer(512, 256, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.score_map = ConvLayer(256, classes, kernel_size=3, stride=1)

    def forward(self, x, fa_points):
        input = x
        down_sample_ratio = 4
        mid_image_size = [
            int(x.size(2) / down_sample_ratio),
            int(x.size(3) / down_sample_ratio)
        ]
        # get attention map
        attn_map = self.attn(x)
        attn_map = F.upsample_bilinear(attn_map, mid_image_size)

        output = []
        fa_points = fa_points.transpose(0, 1)
        for fa_point in fa_points:
            x = self.warp_distort(input, fa_point)
            x = self.seg(x)
            x = self.un_warp_distort(x, fa_point / down_sample_ratio,
                                     mid_image_size)

            merge = attn_map + x
            x = self.conv1(merge)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.score_map(x)
            output.append(x)
        return output

    def warp_distort(self,
                     x,
                     fa_point,
                     out_size=(512, 512),
                     rescale_factor=0.99):
        out_h, out_w = out_size
        batch_size, in_c, in_h, in_w = x.size()
        if self.debug:
            print('warp distort input size', x.size())
            print('warp distort fa_point size', fa_point.size())

        # generate mesh grid
        x_range = torch.tensor(range(out_h)).to(torch.float)
        y_range = torch.tensor(range(out_w)).to(torch.float)
        map_x, map_y = torch.meshgrid([x_range, y_range])

        map_y = torch.unsqueeze(map_y, dim=0).repeat([batch_size, 1, 1])
        map_y = map_y / out_h
        map_x = torch.unsqueeze(map_x, dim=0).repeat([batch_size, 1, 1])
        map_x = map_x / out_w

        # calculate crop box on image
        min_x = torch.min(fa_point[:, :, 0], 1)[0]
        max_x = torch.max(fa_point[:, :, 0], 1)[0]
        min_y = torch.min(fa_point[:, :, 1], 1)[0]
        max_y = torch.max(fa_point[:, :, 1], 1)[0]
        box_w = max_x - min_x
        box_h = max_y - min_y
        box_cx = (max_x + min_x) / 2
        box_cy = (max_y + min_y) / 2
        x1 = box_cx - box_w
        y1 = box_cy - box_h * 3 / 2
        x2 = box_cx + box_w
        y2 = box_cy + box_h
        x1, y1, x2, y2 = x1 / in_w, y1 / in_h, x2 / in_w, y2 / in_h

        y1 = y1.view(-1, 1, 1).repeat([1, out_h, out_w])
        x1 = x1.view(-1, 1, 1).repeat([1, out_h, out_w])
        y2 = y2.view(-1, 1, 1).repeat([1, out_h, out_w])
        x2 = x2.view(-1, 1, 1).repeat([1, out_h, out_w])

        # apply distort
        cy, cx = (y1 + y2) * 0.5, (x1 + x2) * 0.5
        dy, dx = (y2 - y1) * 0.5, (x2 - x1) * 0.5
        ratio_y = self.torch_acrtanh((map_y - 0.5) / 0.5 * rescale_factor)
        ratio_y = (ratio_y.cuda() / rescale_factor * dy + cy) * 2. - 1.
        ratio_x = self.torch_acrtanh((map_x - 0.5) / 0.5 * rescale_factor)
        ratio_x = (ratio_x.cuda() / rescale_factor * dx + cx) * 2. - 1.

        return F.grid_sample(
            x, torch.stack((ratio_y, ratio_x), -1), mode='bilinear')

    def un_warp_distort(self,
                        x,
                        fa_point,
                        out_size=(1024, 1024),
                        rescale_factor=0.99):
        out_h, out_w = out_size
        batch_size, in_c, in_h, in_w = x.size()
        if self.debug:
            print('warp distort input size', x.size())
            print('warp distort fa_point size', fa_point.size())

        # generate mesh grid
        x_range = torch.tensor(range(out_h)).to(torch.float)
        y_range = torch.tensor(range(out_w)).to(torch.float)
        map_x, map_y = torch.meshgrid([x_range, y_range])

        map_y = torch.unsqueeze(map_y, dim=0).repeat([batch_size, 1, 1])
        map_y = map_y / out_h
        map_x = torch.unsqueeze(map_x, dim=0).repeat([batch_size, 1, 1])
        map_x = map_x / out_w

        # calculate crop box on image
        min_x = torch.min(fa_point[:, :, 0], 1)[0]
        max_x = torch.max(fa_point[:, :, 0], 1)[0]
        min_y = torch.min(fa_point[:, :, 1], 1)[0]
        max_y = torch.max(fa_point[:, :, 1], 1)[0]
        box_w = max_x - min_x
        box_h = max_y - min_y
        box_cx = (max_x + min_x) / 2
        box_cy = (max_y + min_y) / 2
        x1 = box_cx - box_w
        y1 = box_cy - box_h * 3 / 2
        x2 = box_cx + box_w
        y2 = box_cy + box_h
        x1, y1, x2, y2 = x1 / out_w, y1 / out_h, x2 / out_w, y2 / out_h

        y1 = y1.view(-1, 1, 1).repeat([1, out_h, out_w])
        x1 = x1.view(-1, 1, 1).repeat([1, out_h, out_w])
        y2 = y2.view(-1, 1, 1).repeat([1, out_h, out_w])
        x2 = x2.view(-1, 1, 1).repeat([1, out_h, out_w])

        # apply distort
        cy, cx = (y1 + y2) * 0.5, (x1 + x2) * 0.5
        dy, dx = (y2 - y1) * 0.5, (x2 - x1) * 0.5
        ratio_y = torch.tanh((map_y.cuda() - cy) / dy * rescale_factor)
        ratio_y = ratio_y / rescale_factor
        ratio_x = torch.tanh((map_x.cuda() - cx) / dx * rescale_factor)
        retio_x = ratio_x / rescale_factor

        return F.grid_sample(
            x, torch.stack((ratio_y, ratio_x), -1), mode='bilinear')

    def torch_acrtanh(self, x, eps=1e-6):
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5


class SegBranch(nn.Module):
    def __init__(self, in_channels, out_channels, debug, back_bone):
        super(SegBranch, self).__init__()
        self.debug = debug
        self.conv1 = ConvLayer(
            in_channels, out_channels, kernel_size=3, stride=1)
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
        if self.debug:
            print('SegBranch input size ', x.size())
        x = self.conv1(x)
        x = self.pool1(x)
        x_1 = self.res_1(x)
        x_2 = self.res_2(x_1)
        x_3 = self.res_3(x_2)
        x_4 = self.res_4(x_3)
        if self.debug:
            print('SegBranch output size', x_4.size())
        return x_4  # [n , 512 , 32 , 32]


class AttnBranch(nn.Module):
    def __init__(self, in_channels, out_channels, debug, back_bone):
        super(AttnBranch, self).__init__()
        self.debug = debug
        attn_feature_size = 128

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
        self.down_channel_attention = ConvLayer(
            2048 // self.expand, attn_feature_size, kernel_size=1, stride=1)
        self.RM = Spatial_Attn(attn_feature_size, attn_feature_size // 8)
        self.up_channel_attention = ConvLayer(
            attn_feature_size, 2048 // self.expand, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x_1 = self.res_1(x)
        x_2 = self.res_2(x_1)
        x_3 = self.res_3(x_2)
        x_4 = self.res_4(x_3)
        if self.debug:
            print('Attn Branch resnet output size', x_4.size())
        x = self.down_channel_attention(x_4)
        x = self.RM(x)
        x_out = self.up_channel_attention(x)
        if self.debug:
            print('Attn Branch ouput size', x_out.size())
        return x_out  # [b , 512 , 32 , 32]
