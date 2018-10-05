import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

#__all__ = ['Spatial_Attn', 'Channel_Attn']


class Spatial_Attn(nn.Module):
    ''' apply spatial aware attention '''

    def __init__(self, in_dim, dim_k=None, out_dim=None):
        super(Spatial_Attn, self).__init__()
        self.channel_in = in_dim
        self.channel_out = out_dim if out_dim is not None else in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8 if dim_k is None else dim_k,
            kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8 if dim_k is None else dim_k,
            kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, width, height = x.size()
        # [b, hxw, C]
        proj_query = self.query_conv(x).view(m_batchsize, -1,
                                             width * height).permute(0, 2, 1)
        # [b, C, hxw]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # [b, hxw, hxw]
        energy = torch.bmm(proj_query, proj_key)
        # [b, hxw, hxw]
        attention = self.softmax(energy)
        # [b, C, hxw]
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # [b, C, hxw]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # [b, C, w, h]
        out = out.view(m_batchsize, C, width, height)
        # [b, C, w, h]
        out = self.gamma * out

        if self.channel_in == self.channel_out:
            out = out + x
        return out


class Channel_Attn(nn.Module):
    ''' apply channel aware attention 
    in_dim: the input image dimention
    '''

    def __init__(self, in_dim, dim_c):
        super(Channel_Attn, self).__init__()
        assert in_dim > 0
        print(f'*************channel {dim_c}')
        self.channel_in = in_dim
        self.channel_out = dim_c
        self.conv_down = nn.Conv2d(
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            kernel_size=1)
        self.conv_up = nn.Conv2d(
            in_channels=self.channel_out,
            out_channels=self.channel_in,
            kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_down(x)
        m_batchsize, C, width, height = x.size()
        # [b, C, hxw ]
        proj_query = x.view(m_batchsize, C, -1)
        # [b, hxw, C]
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # [b, C, C]
        energy = torch.bmm(proj_query, proj_key)
        # [b, C, C]
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # [b, C, C]
        attention = self.softmax(energy_new)
        # [b, C, hxw]
        proj_value = x.view(m_batchsize, C, -1)
        # [b, C, hxw]
        out = torch.bmm(attention, proj_value)
        # [b, C, h, W]
        out = out.view(m_batchsize, C, width, height)
        # [b, C, h, W]
        out = self.gamma * out + x
        out = self.conv_up(out)
        return out


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


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class Multi_Self_Attn(nn.Module):
    """apply multi self attention, based on self attention"""

    def __init__(self, feature_size, dim_k=None, attn_nb=16):
        super(Multi_Self_Attn, self).__init__()
        if not feature_size % attn_nb == 0:
            raise "feature_size cant't be divide by attention number"

        dim_k = feature_size // 8 if dim_k is None else dim_k
        self.attn_nb = attn_nb
        self.attn_list = []
        for i in range(attn_nb):
            self.attn_list.append(
                Self_Attn(
                    feature_size, dim_k, feature_size_out=feature_size // 16))
        self.attn_list = nn.ModuleList(self.attn_list)

    def forward(self, x):
        """
            inputs: 
                x : B x C x M x N
            returns:
                out: B x C x M x N 
        """
        attn_outs = []
        for i in range(self.attn_nb):
            attn_outs.append(
                self.attn_list[i](x))  # B x C//attn_nb x M x N for each out
        out = torch.cat(attn_outs, dim=1)
        return out + x


class Dual_Attn(nn.Module):
    """dual attention module that is Spatial Attention and Channel Attention"""

    def __init__(self, feature_size, dim_k=None):
        super(Dual_Attn, self).__init__()
        dim_k = feature_size // 8 if dim_k is None else dim_k
        dim_c = dim_k
        self.spatial_attn = Spatial_Attn(feature_size, dim_k)
        self.channel_attn = Channel_Attn(feature_size, dim_c)

    def forward(self, x):
        spatial_out = self.spatial_attn(x)
        channel_out = self.channel_attn(x)
        return spatial_out + channel_out


# --------------------- misc -----------------------
class Self_Attn(nn.Module):
    """ Self attention Layer , scratch from github , use 1x1_conv as fc"""

    def __init__(self, feature_size, dim_k, feature_size_out=None):
        super(Self_Attn, self).__init__()
        self.channel_out = feature_size_out if feature_size_out is not None else feature_size
        self.channel_in = feature_size
        self.dim_k = dim_k

        self.query_conv = nn.Conv2d(
            in_channels=self.channel_in, out_channels=dim_k, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=self.channel_in, out_channels=dim_k, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : 
                    self attention value + input feature , B x C x W x H
                    or
                    self attention value , B x Channel_out x W x H
                    depend on if the value of the feature_size_out equal to feature_size
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,
                                             -1, width * height).permute(
                                                 0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1,
                                         width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1,
                                             width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 1, 2))
        out = out.view(m_batchsize, -1, width, height)

        out = self.gamma * out

        if self.channel_in == self.channel_out:
            out = out + x
        return out  # , attention
