import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class DFN(nn.Module):
    def __init__(self,
                 in_channels=5,
                 out_channels=64,
                 add_fc=True,
                 self_attention=False,
                 debug=False,
                 back_bone='resnet101'):
        super(DFN, self).__init__()
        self.add_fc = add_fc  # if flatten and fc the last stage
        self.self_attention = self_attention  # if add self attention
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
            2048 / self.expand, 128, kernel_size=1,
            stride=1)  # choose 128 or 512
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # for fc
        if self.add_fc:
            self.down_channel = ConvLayer(
                2048 / self.expand, 128, kernel_size=1, stride=1)
            self.fc1 = ConvLayer(
                128 * 16 * 16, 1024 * 2, kernel_size=1, stride=1)
            self.fc2 = ConvLayer(
                1024 * 2, 512 * 8 * 8, kernel_size=1, stride=1)

        # for self_attention
        if self.self_attention:
            feature_size = 512
            dim_k = feature_size // 8
            self.down_channel_attention = ConvLayer(
                2048 / self.expand, feature_size, kernel_size=3, stride=2)
            #self.RM = RelationModule(feature_size, dim_k)
            #self.RM = Self_Attn(feature_size, dim_k)
            self.RM = Multi_Self_Attn(feature_size , dim_k)

        self.stage_1 = StageBlock(1, self.expand)
        self.score_map_1 = ConvLayer(512, 2, kernel_size=1, stride=1)
        self.stage_2 = StageBlock(2, self.expand)
        self.score_map_2 = ConvLayer(512, 2, kernel_size=1, stride=1)
        self.stage_3 = StageBlock(3, self.expand)
        self.score_map_3 = ConvLayer(512, 2, kernel_size=1, stride=1)
        self.stage_4 = StageBlock(4, self.expand)
        self.score_map_4 = ConvLayer(512, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x_ = x
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
        #F.upsample(x, x_.size()[2:], mode='bilinear'),
        return [score_1, score_2, score_3, score_4]


class RelationModule(nn.Module):
    def __init__(self, feature_size, dim_k):
        super(RelationModule, self).__init__()
        self.feature_size = feature_size
        self.dim_k = dim_k
        self.W_v = nn.Parameter(torch.Tensor(dim_k, feature_size))
        self.W_q = nn.Parameter(torch.Tensor(dim_k, feature_size))
        self.W_k = nn.Parameter(torch.Tensor(dim_k, feature_size))
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)

    def forward(self, x):
        '''
        x: shape [b , c , h , w]
        return: shape [b ,c , h,w]
        '''
        x_ = x
        x_shape = x.size()
        x_t = x.view(x_shape[0], x_shape[1], -1).contiguous()  # [b , c, h * w]
        Q = torch.bmm(
            self.W_q.unsqueeze(0).expand(x_shape[0], self.dim_k,
                                         self.feature_size), x_t)
        Q = torch.transpose(Q, 1, 2)  # [ b , sample_num , dim_k]
        V = torch.bmm(
            self.W_v.unsqueeze(0).expand(x_shape[0], self.dim_k,
                                         self.feature_size), x_t)
        V = torch.transpose(V, 1, 2)  # [b , sample_num , dim_k]
        K = torch.bmm(
            self.W_k.unsqueeze(0).expand(x_shape[0], self.dim_k,
                                         self.feature_size), x_t)
        W = F.softmax(
            torch.bmm(Q, K) / (self.dim_k**0.5),
            dim=2)  # [b , sample_num , sample_num]
        out = torch.bmm(W, V)  # [b , sample_num , dim_k]
        out = torch.transpose(out, 1, 2)
        out = out.view(x_shape[0], self.dim_k, x_shape[2], x_shape[3])
        # print('Q.shape:', Q.size())
        # print('K.shape:', K.size())
        # print('V.shape:', V.size())
        # print('W.shape:', W.size())
        # print('out.shape:', out.size())
        return x_ + out


class StageBlock(nn.Module):
    def __init__(self, stage=1, expand=1):
        super(StageBlock, self).__init__()
        assert stage in [1, 2, 3, 4]
        if stage == 1:
            in_channels = 256 / expand
        elif stage == 2:
            in_channels = 512 / expand
        elif stage == 3:
            in_channels = 1024 / expand
        elif stage == 4:
            in_channels = 2048 / expand

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
