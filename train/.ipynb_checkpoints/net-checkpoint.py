#!/usr/bin/python3
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log,ceil,floor

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.MaxPool2d):
            pass
        elif isinstance(m, nn.Upsample):
            pass
        elif isinstance(m, nn.LayerNorm):
            pass
        elif isinstance(m, nn.Conv1d):
            pass
        elif isinstance(m, nn.Softmax):
            pass
        else:
            m.initialize()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1      = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1    = nn.ReLU()
        self.fc2      = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out     = avg_out + max_out
        return self.sigmoid(out)

    def initialize(self):
        weight_init(self)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    def initialize(self):
        weight_init(self)

class Res_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

    def initialize(self):
        weight_init(self)

class AOCL(nn.Module):
    def __init__(self, channel, n=2, b=2, inter='ceil'):
        super(AOCL, self).__init__()

        kernel_size = int(abs((log(channel, 2) + 1) / 2))
        self.kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.layer_size = 1 + (log(channel, 2) - b) / (2 * n)
        assert inter == 'ceil' or 'floor'
        if inter == 'ceil':
            self.layer_size = ceil(self.layer_size)
        else:
            self.layer_size = floor(self.layer_size)

        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=int(self.kernel_size / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h = self.conv(x.squeeze(-1).transpose(-1, -2))
        for i in range(self.layer_size):
            h = self.conv(h)
        h = h.transpose(-1, -2).unsqueeze(-1)

        h = self.sigmoid(h)
        return x + h
    def initialize(self):
        weight_init(self)

class AODF(nn.Module):

    def __init__(self, in_channel,out_channel,kernel_size=1, stride=1):
        super(AODF, self).__init__()

        self.in_channel = in_channel
        self.inter_channel = in_channel // 2
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        ratio = 4

        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(self.inter_channel, 1, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.inter_channel // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_channel // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channel // ratio, self.in_channel, kernel_size=1),
            nn.LayerNorm([self.in_channel, 1, 1]),

        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True))
        self.AOCL = AOCL(self.inter_channel)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.out = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True)
        )

    def channel(self, E, D):
        C = E
        D_ = self.conv_2(D)
        batch, channel, height, width = D_.size()
        D_ = D_.view(batch, channel, height * width)  # [N, IC, H*W]

        E = self.conv_1(E)  # [N, 1, H, W]
        E = E.view(batch, 1, height * width)  # [N, 1, H*W]
        E = self.softmax(E)  # [N, 1, H*W]

        context = torch.matmul(D_, E.transpose(1, 2)).unsqueeze(-1)  # [N, IC, 1, 1]
        context = self.conv_up(self.AOCL(context))  # [N, OC, 1, 1]

        out = D * self.sigmoid(context)+C  # [N, OC, 1, 1]
        return out

    def spatial(self, E, D):
        C = E
        E_ = self.conv_3(E)  # [N, IC, H, W]
        batch, channel, height, width = E_.size()
        E_ = self.AOCL(self.avg_pool(E_))  # [N, IC, 1, 1]
        batch, channel, avg_e_h, avg_e_w = E_.size()
        E_ = E_.view(batch, channel, avg_e_h * avg_e_w).permute(0, 2, 1)  # [N, 1, IC]
        E_ = self.softmax(E_)

        D = self.conv_4(D).view(batch, self.inter_channel, height * width)  # [N, IC, H*W]

        context = torch.matmul(E_, D).view(batch, 1, height, width)  # [N, 1, H, W]
        context = F.layer_norm(context, normalized_shape=(1, context.shape[-2], context.shape[-1]))

        out = E * self.sigmoid(context)+C  # [N, 1, H, W]
        return out

    def forward(self, E, D):
        channel = self.channel(E, D)  # [N, C, H, W]
        spatial = self.spatial(E, D)  # [N, C, H, W]
        out = self.out(spatial + channel)  # [N, C, H, W]
        return out
    def initialize(self):
        weight_init(self)

class CDC_CA_CB_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1,
                 padding=1, dilation=1, theta=0.7, padding_mode='zeros', name = 'ca'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                              stride=stride, dilation=dilation, bias=bias, padding_mode=padding_mode)
        self.theta = theta
        self.name = name

    def forward(self, x):
        norm_out = self.conv(x)
        if (self.theta - 0.0) < 1e-6:
            return norm_out
        else:
            if self.name == 'ca':
                central_mask = torch.tensor([[0, 1, 0],
                                             [1, -4, 1],
                                             [0, 1, 0]], dtype=torch.float32, device=x.device)
            else:
                central_mask = torch.tensor([[1, 0, 1],
                                             [0, -4, 0],
                                             [1, 0, 1]], dtype=torch.float32, device=x.device)
            central_mask = central_mask.repeat(self.conv.out_channels, self.conv.in_channels, 1, 1)

            # Convert central_mask to the same data type as the input tensor
            central_mask = central_mask.type_as(x)

            # Get stride and padding parameters
            stride = self.conv.stride
            padding = self.conv.padding
            if isinstance(stride, tuple) or isinstance(stride, list):
                stride = stride[0]
            if isinstance(padding, tuple) or isinstance(padding, list):
                padding = padding[0]

            # Adjust padding to match the original convolution's output size
            input_padded = F.pad(x, [padding, padding, padding, padding], mode='replicate')
            diff_out = F.conv2d(input=input_padded, weight=central_mask, bias=None, stride=stride, padding=0)

            # Combine standard convolution output and central difference output
            out = norm_out - self.theta * diff_out
            return out
    
    def initialize(self):
        weight_init(self)

class CDC_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1,
                padding=1, dilation=1, theta=0.7, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            stride = stride, dilation=dilation, bias=bias, padding_mode=padding_mode)
        self.theta = theta
        self.CDC_CA = nn.Sequential(
            CDC_CA_CB_conv(in_channels, out_channels, kernel_size=3, padding=1, theta=theta, padding_mode=padding_mode,
            stride=stride, bias=bias, name = 'ca'),
            nn.BatchNorm2d(out_channels),
        )
        self.CDC_CB = nn.Sequential(
            CDC_CA_CB_conv(in_channels, out_channels, kernel_size=3, padding=1, theta=theta, padding_mode=padding_mode,
            stride=stride, bias=bias, name = 'cb'),
            nn.BatchNorm2d(out_channels),
        )
        self.conv1x1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        norm_out = self.conv(x)
        if (self.theta - 0.0) < 1e-6:
            return norm_out
        else:
            ca = self.CDC_CA(x)
            cb = self.CDC_CB(x)
            out = self.conv1x1(torch.cat((ca, cb), dim=1))
            return out
    def initialize(self):
        weight_init(self)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, theta_1=0, theta_2=0.7, theta_r=0., norm=nn.BatchNorm2d,
                 padding_mode='zeros'):
        super().__init__()
        self.conv_block = nn.Sequential(
            CDC_conv(in_c, out_c, kernel_size=3, padding=1, theta=theta_1, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
            nn.ReLU(inplace=True),
            CDC_conv(out_c, out_c, kernel_size=3, padding=1, theta=theta_2, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.residual_block = nn.Sequential(
            CDC_conv(in_c, out_c, kernel_size=3, padding=1, theta=theta_r, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or out_c != in_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        conv_out = self.conv_block(x)
        residual_out = self.residual_block(x)
        out = self.relu(conv_out + residual_out + residual)
        return out
    def initialize(self):
        weight_init(self)

class Cdcbone(nn.Module):
    def __init__(self, block,  nb_filter, block_nums):
        super(Cdcbone, self).__init__()
        input_channel = 3
        self.pool    = nn.MaxPool2d(2, 2)
        self.conv0_0 = self._make_layer(block, input_channel, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], block_nums[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], block_nums[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], block_nums[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], block_nums[3])
        
        self.norm = nn.BatchNorm2d
        theta_0=0.9
        theta_1=0
        theta_2=0.9
        theta_r=0
        self.conv1 = nn.Sequential(
            CDC_conv(3, nb_filter[0], bias=False, theta=theta_0),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            CDC_conv(nb_filter[0], nb_filter[0], bias=False, theta=theta_0),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(nb_filter[0],  nb_filter[1], norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=1)
        self.layer2 = ResidualBlock(nb_filter[1], nb_filter[2], norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=1)
        self.layer3 = ResidualBlock(nb_filter[2], nb_filter[3], norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=1)
        self.layer4 = ResidualBlock(nb_filter[3], nb_filter[4], norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=1)

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv1(input)
        x0_0 = self.conv2(x0_0)
        x1_0 = self.layer1(self.pool(x0_0))
        x2_0 = self.layer2(self.pool(x1_0))
        x3_0 = self.layer3(self.pool(x2_0))
        x4_0 = self.layer4(self.pool(x3_0))
        return x0_0, x1_0, x2_0, x3_0, x4_0

    def initialize(self):
        weight_init(self)


class Decoder1(nn.Module):
    def __init__(self, block,  nb_filter, block_nums):
        super(Decoder1, self).__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1], nb_filter[1], block_nums[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2], nb_filter[2], block_nums[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3], nb_filter[3], block_nums[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1], nb_filter[1], block_nums[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2], nb_filter[2], block_nums[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1], nb_filter[1], block_nums[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv_1_0_en = nn.Conv2d(16, 16, kernel_size=1)
        self.conv_1_0_de = nn.Conv2d(32, 16, kernel_size=1)
        self.conv_1_1_en = nn.Conv2d(64, 32, kernel_size=1)
        self.conv_1_1_de = nn.Conv2d(64, 32, kernel_size=1)
        self.conv_2_1_en = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_2_1_de = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_3_1_en = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_3_1_de = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_0_2_en = nn.Conv2d(64, 16, kernel_size=1)
        self.conv_0_2_de = nn.Conv2d(32, 16, kernel_size=1)
        self.conv_1_2_en = nn.Conv2d(160, 32, kernel_size=1)
        self.conv_1_2_de = nn.Conv2d(64, 32, kernel_size=1)
        self.conv_2_2_en = nn.Conv2d(336, 64, kernel_size=1)
        self.conv_2_2_de = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_0_3_en = nn.Conv2d(80, 16, kernel_size=1)
        self.conv_0_3_de = nn.Conv2d(32, 16, kernel_size=1)
        self.conv_1_3_en = nn.Conv2d(192, 32, kernel_size=1)
        self.conv_1_3_de = nn.Conv2d(64, 32, kernel_size=1)
        self.conv_0_4_en = nn.Conv2d(160, 16, kernel_size=1)
        self.conv_0_4_de = nn.Conv2d(32, 16, kernel_size=1)
        
        self.AODF_1 = AODF(16,16)
        self.AODF_2 = AODF(32,32)
        self.AODF_3 = AODF(64,64)
        self.AODF_4 = AODF(128,128)
        self.AODF_5 = AODF(16,16)
        self.AODF_6 = AODF(32,32)
        self.AODF_7 = AODF(64,64)
        self.AODF_8 = AODF(16,16)
        self.AODF_9 = AODF(32,32)
        self.AODF_10 = AODF(16,16)

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = input[0]
        x1_0 = input[1]
        x2_0 = input[2]
        x3_0 = input[3]
        x4_0 = input[4]
        
        AODF_1 = self.AODF_1(self.conv_1_0_de(self.up(x1_0)), self.conv_1_0_en(x0_0))
        AODF_1 = AODF_1 +  self.conv_1_0_en(x0_0)
        x0_1 = self.conv0_1(AODF_1)
        AODF_2 = self.AODF_2(self.conv_1_1_de(self.up(x2_0)), self.conv_1_1_en(torch.cat([x1_0, self.down(x0_0), self.down(x0_1)], 1)))
        AODF_2 = AODF_2 +  self.conv_1_1_en(torch.cat([x1_0, self.down(x0_0), self.down(x0_1)], 1))
        x1_1 = self.conv1_1(AODF_2)
        AODF_3 = self.AODF_3(self.conv_2_1_de(self.up(x3_0)), self.conv_2_1_en(torch.cat([x2_0, self.down(x1_0), self.down(x1_1)], 1)))
        AODF_3 = AODF_3 +  self.conv_2_1_en(torch.cat([x2_0, self.down(x1_0), self.down(x1_1)], 1))
        x2_1 = self.conv2_1(AODF_3)
        AODF_4 = self.AODF_4(self.conv_3_1_de(self.up(x4_0)), self.conv_3_1_en(torch.cat([x3_0, self.down(x2_0), self.down(x2_1)], 1)))
        AODF_4 = AODF_4 +  self.conv_3_1_en(torch.cat([x3_0, self.down(x2_0), self.down(x2_1)], 1))
        x3_1 = self.conv3_1(AODF_4)

        AODF_5 = self.AODF_5(self.conv_0_2_de(self.up(x1_1)), self.conv_0_2_en(torch.cat([x0_0, x0_1, self.up(x1_0)], 1)))
        AODF_5 = AODF_5 +  self.conv_0_2_en(torch.cat([x0_0, x0_1, self.up(x1_0)], 1))
        x0_2 = self.conv0_2(AODF_5)
        AODF_6 = self.AODF_6(self.conv_1_2_de(self.up(x2_1)), self.conv_1_2_en(torch.cat([x1_0, x1_1, self.down(x0_1), self.down(x0_2), self.up(x2_0)], 1)))
        AODF_6 = AODF_6 +  self.conv_1_2_en(torch.cat([x1_0, x1_1, self.down(x0_1), self.down(x0_2), self.up(x2_0)], 1))
        x1_2 = self.conv1_2(AODF_6)
        AODF_7 = self.AODF_7(self.conv_2_2_de(self.up(x3_1)), self.conv_2_2_en(torch.cat([x2_0, x2_1, self.down(x1_1), self.down(x1_2), self.down(self.down(x0_0)), self.up(x3_0)], 1)))
        AODF_7 = AODF_7 +  self.conv_2_2_en(torch.cat([x2_0, x2_1, self.down(x1_1), self.down(x1_2), self.down(self.down(x0_0)), self.up(x3_0)], 1))
        x2_2 = self.conv2_2(AODF_7)

        AODF_8 = self.AODF_8(self.conv_0_3_de(self.up(x1_2)), self.conv_0_3_en(torch.cat([x0_0, x0_1, x0_2, self.up(x1_1)], 1)))
        AODF_8 = AODF_8 +  self.conv_0_3_en(torch.cat([x0_0, x0_1, x0_2, self.up(x1_1)], 1))
        x0_3 = self.conv0_3(AODF_8)
        AODF_9 = self.AODF_9(self.conv_1_3_de(self.up(x2_2)), self.conv_1_3_en(torch.cat([x1_0, x1_1, x1_2, self.down(x0_2), self.down(x0_3), self.up(x2_1)], 1)))
        AODF_9 = AODF_9 +  self.conv_1_3_en(torch.cat([x1_0, x1_1, x1_2, self.down(x0_2), self.down(x0_3), self.up(x2_1)], 1))
        x1_3 = self.conv1_3(AODF_9)
        
        AODF_10 = self.AODF_10(self.conv_0_4_de(self.up(x1_3)), self.conv_0_4_en(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_2), self.up_4(x2_0)], 1)))
        AODF_10 = AODF_10 +  self.conv_0_4_en(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_2), self.up_4(x2_0)], 1))
        x0_4 = self.conv0_4(AODF_10)
        return x4_0, x3_1, x2_2, x1_3, x0_4

    def initialize(self):
        weight_init(self)


class Decoder2(nn.Module):
    def __init__(self, block, nb_filter, block_nums):
        super(Decoder2, self).__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4  = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8  = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])
        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x4_0 = input[0]
        x3_1 = input[1]
        x2_2 = input[2]
        x1_3 = input[3]
        x0_4 = input[4]

        fx4_0 = self.up_16(self.conv0_4_1x1(x4_0))
        fx3_1 = self.up_8(self.conv0_3_1x1(x3_1))
        fx2_2 = self.up_4(self.conv0_2_1x1(x2_2))
        fx1_3 = self.up(self.conv0_1_1x1(x1_3))
        fx0_4 = x0_4

        Final_x0_4 = self.conv0_4_final(torch.cat([fx4_0, fx3_1, fx2_2, fx1_3, fx0_4], 1))
        return fx4_0, fx3_1, fx2_2, fx1_3, fx0_4, Final_x0_4

    def initialize(self):
        weight_init(self)


class Decoder3(nn.Module):
    def __init__(self, block, nb_filter, block_nums):
        super(Decoder3, self).__init__()
        self.conv_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])
        self.conv4_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.conv3_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.conv2_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.conv1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input1):
        out0 = torch.cat([input1[0], input1[0]], dim=1)
        out1 = torch.cat([input1[1], input1[1]], dim=1)
        out2 = torch.cat([input1[2], input1[2]], dim=1)
        out3 = torch.cat([input1[3], input1[3]], dim=1)
        out4 = torch.cat([input1[4], input1[4]], dim=1)

        x0 = self.conv4_1x1(out0)
        x1 = self.conv3_1x1(out1)
        x2 = self.conv2_1x1(out2)
        x3 = self.conv1_1x1(out3)
        x4 = self.conv1_1x1(out4)

        Final_x = self.conv_final(torch.cat([x0, x1, x2, x3, x4], 1))
        return Final_x

    def initialize(self):
        weight_init(self)


class DEFER(nn.Module):
    def __init__(self, cfg, block,  nb_filter, block_nums):
        super(DEFER, self).__init__()
        self.cfg = cfg
        self.bCdcbone = Cdcbone(block,  nb_filter, block_nums)
        self.decoderb1 = Decoder1(block,  nb_filter, block_nums)
        self.decoderb2 = Decoder2(block,  nb_filter, block_nums)
        self.decoder3 = Decoder3(block, nb_filter, block_nums)

        self.linear = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=3, padding=1))
        self.initialize()

    def forward(self, x, shape=None):
        xb0_0, xb1_0, xb2_0, xb3_0, xb4_0 = self.bCdcbone(x)
        xb4_0, xb3_1, xb2_2, xb1_3, xb0_4 = self.decoderb1([xb0_0, xb1_0, xb2_0, xb3_0, xb4_0])
        fxb4_0, fxb3_1, fxb2_2, fxb1_3, fxb0_4, Final_xb0_4 = self.decoderb2([xb4_0, xb3_1, xb2_2, xb1_3, xb0_4])
        out0_4 = self.decoder3([fxb4_0, fxb3_1, fxb2_2, fxb1_3, fxb0_4])
        out0_4 = torch.cat([Final_xb0_4, out0_4], dim=1)

        if shape is None:
            shape = x.size()[2:]
        out4 = F.interpolate(self.linear(out0_4), size=shape, mode='bilinear')
        return out4

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
