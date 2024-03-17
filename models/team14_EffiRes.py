import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import math

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ESA(nn.Module):


    def __init__(self, n_feats, conv, esa_channels = 16):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                            mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
   


class Conv_layer(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size=3, bias = True):
        super(Conv_layer, self).__init__()
        kernel_size = _make_pair(kernel_size)
        padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
        self.conv = nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=kernel_size, padding=padding, bias=bias)
       
    def forward(self, x):
        return self.conv(x)


class RLFB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16, kernel_size = 3):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.attn =  ESA(out_channels, nn.Conv2d)

        self.conv1_r = Conv_layer(in_channels,mid_channels, kernel_size)
        self.conv2_r = Conv_layer(mid_channels,mid_channels, kernel_size)

        self.act = nn.LeakyReLU(0.05)
        self.c5 = Conv_layer(in_channels, out_channels,1)

        self.conv_out = Conv_layer(in_channels*2,in_channels,1)
   
    def forward(self, x):
        out = self.act(self.conv1_r(x))
        out1 = self.act(self.conv2_r(out))

        out = out1 + x + out

        x = self.attn(self.c5(out))
       
        return x
    
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
       
class EffiRes(nn.Module):
    def __init__(self,upscale=4,in_channel= 3, out_channel=3,feature_depth = 36, kernel = 3, pad=1):
        super(EffiRes, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
        self.add_mean = MeanShift(3, rgb_mean, rgb_std, 1)

        conv = default_conv

        modules_tail = [
            Upsampler(conv, upscale, feature_depth, act=False),
            conv(feature_depth,out_channel, kernel)]


        self.head = conv(in_channel, feature_depth, kernel)

        
        self.body_1 = RLFB(feature_depth, kernel_size= kernel)
        self.body_2 = RLFB(feature_depth, kernel_size= kernel)
        self.body_3 = RLFB(feature_depth, kernel_size= kernel)
        self.body_4 = RLFB(feature_depth, kernel_size= kernel)

        self.tail_1 = Upsampler(conv, upscale, feature_depth, act=False)
        self.tail_2 = conv(feature_depth,out_channel, kernel)

        self.body_tail = conv(feature_depth, feature_depth, kernel)

    def forward(self, x):

        x = self.sub_mean(x)
        x = self.head(x)


        res = x
        x = self.body_1(x)
        x = self.body_2(x)
        x = self.body_3(x)
        x = self.body_4(x)


        res =x +res   
        res = self.body_tail(res)

        
        x = self.tail_1(res)
        x = self.tail_2(x)
        x = self.add_mean(x)

       
        return x


