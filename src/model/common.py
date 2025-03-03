import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


def default_conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = True) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


def default_conv1(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = True) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 5), stride=stride, bias=bias)


def weight_init(module: nn.Module):
    """init weight.

    Args:
        module (nn.Module): module of network to be initialized.
    """
    for n, m in module.named_children():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # type : ignore


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized.
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal.
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might.
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class MeanShift(torch.nn.Module):
    def __init__(self, mean, std):
        super(MeanShift, self).__init__()
        self.mean = torch.FloatTensor([mean]).view(1, -1, 1, 1)
        self.std = torch.FloatTensor([std]).view(1, -1, 1, 1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input):
        return (input - self.mean.to(input.device)) / self.std.to(input.device)


class UnNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = torch.FloatTensor([mean]).view(1, -1, 1, 1)
        self.std = torch.FloatTensor([std]).view(1, -1, 1, 1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input):
        return input * self.std.to(input.device) + self.mean.to(input.device)


class UpSampler(nn.Sequential):
    """UpSampling for reconstruction.
    """

    def __init__(self, scale: int, n_feats: int, bn: bool = False, act=False, bias: bool = True):
        """Initialize UpSample module.

        Args:
            scale (int): scale of reconstruction.
            n_feats (int): in_channels and out_channels of this module.
            bn (bool, optional): _description_. Defaults to False.
            act (any, optional): _description_. Defaults to False.
            bias (bool, optional): bias of conv. Defaults to True.

        Raises:
            NotImplementedError: not nn.Sequential.
        """
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(default_conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(UpSampler, self).__init__(*m)


class UpSampler1(nn.Sequential):
    """UpSampling for reconstruction.
    """

    def __init__(self, scale: int, n_feats: int, bn: bool = False, act=False, bias: bool = True):
        """Initialize UpSample module.

        Args:
            scale (int): scale of reconstruction.
            n_feats (int): in_channels and out_channels of this module.
            bn (bool, optional): _description_. Defaults to False.
            act (any, optional): _description_. Defaults to False.
            bias (bool, optional): bias of conv. Defaults to True.

        Raises:
            NotImplementedError: not nn.Sequential.
        """
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(default_conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(UpSampler1, self).__init__(*m)



class ResBlock(nn.Module):
    """ResBlock contains conv, act and conv.
    """

    def __init__(self, n_feats: int, kernel_size: int, bias: bool = True, bn=False, act: nn.Module = nn.ReLU(True)):
        """Initializes internal Module state.

        Args:
            n_feats (int): in_channels and out_channels of this residual block.
            kernel_size (int): kernel size of conv of this residual block.
            bias (bool, optional): conv bias. Defaults to True.
            bn (bool, optional): is or not BatchNorm. Defaults to False.
            act (nn.Module, optional): active function. Defaults to nn.ReLU(True).
        """
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):  # 每个残差模块两个卷积层 一个Relu
            m.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class nResBlock(nn.Module):
    """a lightweight ResBlock
    """

    def __init__(self, n_feats: int, key_feats: int, kernel_size: int, bias: bool = True, bn: bool = False, act: nn.Module = nn.ReLU(True)) -> None:
        """Initializes internal Module state.

        Args:
            n_feats (int): in_channels and out_channels of this residual block.
            key_feats (int): medium_channels of this residual block.
            kernel_size (int): kernel size of conv of this residual block.
            bias (bool, optional): conv bias. Defaults to True.
            bn (bool, optional): is or not BatchNorm. Defaults to False.
            act (nn.Module, optional): active function. Defaults to nn.ReLU(True).
        """
        super().__init__()
        self.body = nn.Sequential(
            default_conv(n_feats, key_feats, 1, bias=True),
            act,
            default_conv(key_feats, key_feats, 3, bias=True),
            act,
            default_conv(key_feats, n_feats, 1, bias=True)
        )
        self.act = act

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class cResBlock(nn.Module):
    """Define a residual block of CycleGAN"""

    def __init__(self, dim: int, padding_type: str, norm_layer, use_dropout: bool, bias):
        """Initialize the residual block

        A residual block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        """
        super(cResBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, bias):
        """Construct a convolution block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=p, bias=bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=p, bias=bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x) -> torch.Tensor:
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class cResGenerator(nn.Module):
    """A residual generator of CycleGAN.
    Residual-based generator that consists of residual blocks between a few down-sampling/up-sampling operations.
    """

    def __init__(self, in_channels: int, out_channels: int, key_channels: int = 64, norm_layer=nn.InstanceNorm2d, use_dropout: bool = False, n_blocks: int = 6, padding_type='reflect', bias: bool = True):
        """Initializes internal Module state.

        Args:
            in_channels (int): in_channels of this ResGenerator block.
            out_channels (int): out_channels of this ResGenerator block.
            key_channels (int, optional): medium filters of block. Defaults to 64.
            norm_layer (_type_, optional): normalization layer. Defaults to nn.InstanceNorm2d.
            use_dropout (bool, optional): need dropout or not. Defaults to False.
            n_blocks (int, optional): number of residual block. Defaults to 6.
            padding_type (str, optional): type of the added padding. Defaults to 'reflect'.
            bias (bool, optional): conv bias. Defaults to True.
        """
        super(cResGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, key_channels,
                           kernel_size=7, padding=0, bias=bias),
                 norm_layer(key_channels),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(key_channels * mult, key_channels * mult * 2, kernel_size=3, stride=2, padding=1, bias=bias),
                      norm_layer(key_channels * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [cResBlock(key_channels * mult, padding_type=padding_type,
                                norm_layer=norm_layer, use_dropout=use_dropout, bias=bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(key_channels * mult, int(key_channels * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=bias),
                      norm_layer(int(key_channels * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(key_channels, out_channels,
                            kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels: int, key_channels: int = 64, n_layers: int = 3, norm_layer=nn.InstanceNorm2d, bias: bool = True):
        """Construct a PatchGAN discriminator

        Args:
            in_channels (int): channels of input tensor
            key_channels (int, optional): medium filters of block. Defaults to 64.
            n_layers (int, optional): the number of conv layers in the discriminator. Defaults to 3.
            norm_layer (_type_, optional): normalization layer. Defaults to nn.InstanceNorm2d.
            bias (bool, optional): conv bias. Defaults to True.
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        n_pad = 1
        sequence = [nn.Conv2d(in_channels, key_channels, kernel_size=kw,
                              stride=2, padding=n_pad), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(key_channels * nf_mult_prev, key_channels * nf_mult,
                          kernel_size=kw, stride=2, padding=n_pad, bias=bias),
                norm_layer(key_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(key_channels * nf_mult_prev, key_channels * nf_mult,
                      kernel_size=kw, stride=1, padding=n_pad, bias=bias),
            norm_layer(key_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(key_channels * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=n_pad)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class ResGenerator(nn.Sequential):
    """ A residual generator of CinGAN.

    A residual generator contains a head block (three convolutions and acts),
    a body block (6 residual blocks) and a tail block 
    (three convolutions and acts, transfer channels of input to target channels of output).
    """
    # distinguish generators from each other by h_stride

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool = True, act: nn.Module = nn.LeakyReLU(0.2, True)) -> None:
        """Initializes internal Module state.

        Args:
            in_channels (int): in_channels of this ResGenerator block.
            out_channels (int): out_channels of this ResGenerator block.
            kernel_size (int): conv kernel.
            stride (int): conv stride.
            bias (bool, optional): conv bias. Defaults to True.
            act (nn.Module, optional): active function. Defaults to nn.LeakyReLU(0.2, True).
        """
        m = [default_conv(in_channels, 64, 7, 1, bias),
             act,
             nn.Conv2d(64, 64, kernel_size, stride, padding=1, bias=bias),
             act,
             default_conv(64, 64, 3, 1, bias),
             act,
             ]
        m.extend([ResBlock(n_feats=64, kernel_size=3, bias=bias, act=act)
                 for _ in range(6)])
        m.extend([default_conv(64, 64, 3, 1, bias),
                  act,
                  default_conv(64, 64, 3, 1, bias),
                  act,
                  default_conv(64, out_channels, 7, 1, bias)])
        super(ResGenerator, self).__init__(*m)


class Discriminator(nn.Module):
    """ A discriminator of CycleGAN.

    A discriminator contains five convolutions and three BatchNorm layer
    """

    def __init__(self, in_channels: int, out_channels: int, h_stride: int, bias: bool = True, act: nn.Module = nn.LeakyReLU(0.2, True)) -> None:
        """Initializes internal Module state.

        Args:
            in_channels (int): in_channels of this discriminator block.
            out_channels (int): out_channels of this discriminator block.
            kernel_size (int): conv kernel.
            h_stride (int): distinguish between 1st step and 2nd step of CinCGAN 
            bias (bool, optional): conv bias. Defaults to True.
            act (nn.Module, optional): active function. Defaults to nn.LeakyReLU(0.2, True).
        """
        super(Discriminator, self).__init__()
        self.body = nn.Sequential(
            default_conv(in_channels, 64, 4, h_stride, bias),
            act,
            default_conv(64, 128, 4, h_stride, False),
            nn.BatchNorm2d(128),
            act,
            default_conv(128, 256, 4, h_stride, False),
            nn.BatchNorm2d(256),
            act,
            default_conv(256, 512, 4, 1, False),
            nn.BatchNorm2d(512),
            act,
            default_conv(512, out_channels, 4, 1, bias)
        )

    def forward(self, x):
        y = self.body(x)
        return y


class Cycle_G1(nn.Module):
    """the 1st Generator of CycleCNN.
    """

    def __init__(self, scale) -> None:
        """
        Args:
            scale (int): input the scale of reconstruction for average pooling.
        """
        super().__init__()
        m_head = [default_conv(1, 64, 3), nn.ReLU(True)]
        m_body = [ResBlock(64, 3) for _ in range(16)]
        m_body.append(default_conv(64, 64, 3))  # type: ignore
        m_tail = [UpSampler(scale, 64)]
        m_tail.append(default_conv(64, 1, 3))  # type: ignore
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


class Cycle_G2(nn.Module):
    """the 2nd Generator of CycleCNN.
    """

    def __init__(self, scale: int) -> None:
        """
        Args:
            scale (int): input the scale of reconstruction for average pooling.
        """
        super().__init__()
        act = nn.ReLU(True)
        m_head = [default_conv(1, 64, 3), act]
        down_sample = [nn.AvgPool2d(scale), default_conv(64, 64, 3), act]
        m_head.extend(down_sample)
        m_body = [ResBlock(64, 3) for _ in range(5)]
        m_body.append(default_conv(64, 64, 3))  # type: ignore
        m_tail = [default_conv(64, 64, 3), default_conv(64, 1, 3)]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

# RCAN


class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16, act: nn.Module = nn.ReLU(True)) -> None:
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            act,
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int, reduction: int = 16, bias: bool = True, bn=False, act: nn.Module = nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(default_conv(
                n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction, act=act))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x) -> torch.Tensor:
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int, reduction: int, n_resblocks: int) -> None:
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(n_feat, kernel_size, reduction, bias=True,
                 bn=False, act=nn.ReLU(True), res_scale=1)
            for _ in range(n_resblocks)]
        modules_body.append(default_conv(
            n_feat, n_feat, kernel_size))  # type: ignore
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = default_conv(in_channels, out_channels, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate *
                             num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class AMESA(nn.Module):   # 改进后的ESA
    def __init__(self, channel: int, reduction: int = 4) -> None:
        super().__init__()
        act = nn.ReLU(True)
        self.channel_reduction = nn.Sequential(
            default_conv(channel, channel//reduction, 1),
            act,
        )
        conv_group = [default_conv(channel//reduction, channel//reduction, 3), act,
                      default_conv(channel//reduction,
                                   channel//reduction, 3), act,
                      default_conv(channel//reduction, channel//reduction, 3), act,]
        self.conv_stride_pool = nn.Sequential(nn.Conv2d(channel//reduction, channel//reduction, 3, dilation=6, padding=6, bias=True),
                                              act,
                                              *conv_group)
        self.conv_rd = nn.Sequential(default_conv(2 * channel//reduction, channel, 1),
                                     nn.Sigmoid())

    def forward(self, x):
        feature = self.channel_reduction(x)
        feature_conv = self.conv_stride_pool(feature)
        feature_sa = self.conv_rd(torch.cat([feature, feature_conv], dim=1))
        return feature_sa*x                       

class ESA(nn.Module): # 原ESA
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        act = nn.ReLU(True) # 激活函数在卷积层和池化层中可以提取图像的特征
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x) # 对输入使用1x1卷积降通道数 
        c1 = self.conv2(c1_) # 使用stride为2的3*3卷积,降低特征空间尺寸  
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3) # 使用stride为3的7*7的池化核进行最大池化
        # Conv Groups,使用3个3*3卷积,进一步扩大感受野范围
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) # 采用双线性插值法进行上采样得到与输入尺寸大小的特征（张量的第三、四维度为高度、宽度）
        cf = self.conv_f(c1_) # 对c1_另外进行一次1*1卷积
        c4 = self.conv4(c3 + cf) # 跳跃连接并进行一次1*1卷积
        m = self.sigmoid(c4) # 应用sigmoid激活函数,得到注意力权重

        return x * m

class oESA(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(oESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m
    

class Spatial_Attention(nn.Module):

    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        y = x * mask
        return y


class ERes(nn.Module):
    def __init__(self, n_features, ) -> None:
        super().__init__()
        act = nn.ReLU(True)
        self.body = nn.Sequential(
            default_conv(n_features, n_features, 3),
            act,
            default_conv(n_features, n_features, 3),
            ESA(n_features)
        )

    def forward(self, x):
        res = self.body(x)
        return res, res+x


class RFA(nn.Module):
    def __init__(self, n_features) -> None:
        super(RFA, self).__init__()
        self.n_resblocks = 4
        rfa = [ERes(n_features=n_features)
               for _ in range(self.n_resblocks)]
        self.RFA = nn.ModuleList(rfa)
        self.tail = nn.Sequential(
            default_conv(n_features * 4, n_features, 1)
        )

    def forward(self, x):
        feature = x
        res_feature = []
        for index in range(self.n_resblocks):
            tmp, x = self.RFA[index](x)
            res_feature.append(tmp)
        res = self.tail(torch.cat(res_feature, dim=1))
        res += feature
        return res


class SA_Adapt(nn.Module):
    def __init__(self, channels: int):
        super(SA_Adapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_Conv(channels, channels, 3, 1, 1)

    def forward(self, x, scale):
        mask = self.mask(x)
        adapted = self.adapt(x, scale, scale)

        return x + adapted * mask


class SA_Conv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = False, num_experts: int = 4):
        super(SA_Conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(
                channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(
                torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale_h: int, scale_w: int):
        # generate routing weights
        scale: torch.Tensor = torch.ones(1, 1).to(x.device) / scale_h
        scale2: torch.Tensor = torch.ones(1, 1).to(x.device) / scale_w
        routing_weights: torch.Tensor = self.routing(
            torch.cat((scale, scale2), 1)).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(
            self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in,
                                         self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias,
                       stride=self.stride, padding=self.padding)

        return out


class SA_upsample(nn.Module):
    def __init__(self, scale: int, channels: int, num_experts: int = 4, bias: bool = False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.scale = scale
        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(
                torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(
                torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, h, w = x.size()
        scale = scale2 = self.scale
        # (1) coordinates in LR space
        # coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        # coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - \
            (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - \
            (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand(
                [-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand(
                [-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)

        # (2) predict filters and offsets
        embedding = self.body(input)
        # offsets
        offset = self.offset(embedding)

        # filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.num_experts, round(
            scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(
            1, round(scale*h), round(scale2*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(
            1, round(scale*h), round(scale2*w), self.channels, self.channels//8)

        # (3) grid sample & spatially varying filtering
        # grid sample
        fea0 = grid_sample(x, offset, scale, scale2)  # b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)  # b * h * w * c * 1

        # spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand(
            [b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0


def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1), 1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros')

    return output


if __name__ == '__main__':
    x = torch.randn(16, 64, 48, 48)
    model = oESA(64)
    y = model(x)
    print(y.shape)
