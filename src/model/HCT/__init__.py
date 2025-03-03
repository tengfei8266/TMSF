# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/06/10 11:22:40
@Author  :   ztf 
@Contact :   tengfeizhang@whu.edu.cn
@License :   (C)Copyright 2024, Wuhan University
@Desc    :   None
'''

# here put the import lib

import torch
import torch.nn as nn
from src.model.common import AMESA, ESA, default_conv, UpSampler, Spatial_Attention
from src.model.HCT.swin_transformer import SwinTransformer

class ATCB(nn.Module):
    def __init__(self, in_channels: int, n_feats: int, depth: int = 2, window_size: int = 8, num_modules: int = 4) -> None:
        super().__init__()
        m_core_list = [self.ablation_2(
            in_channels, n_feats, depth, window_size) for _ in range(num_modules)]
        # m_core_list = [self.ablation_2(in_channels, n_feats, depth, window_size) for _ in range(
        #     num_modules)]  # Transformer vs Conv
        self.fusion = nn.Sequential(default_conv(
            n_feats * num_modules, n_feats, 1))
        self.core = nn.ModuleList(m_core_list)
        self.num_modules = num_modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = []
        for i in range(self.num_modules):
            x = self.core[i](x)
            local_features.append(x)
        out_fused = self.fusion(torch.cat(local_features, 1))
        return out_fused

    def create_ctb(self, in_channels: int, n_feats: int, depth: int, window_size: int):
        """这里需要测试ESA的重要性(去掉)

        Args:
            in_channels (int): _description_
            n_feats (int): _description_
            depth (int): _description_
            window_size (int): _description_

        Returns:
            _type_: _description_
        """
        m_core_list = [
            AMESA(in_channels),
            SwinTransformer(n_feats=n_feats,
                            depth=depth,
                            window_size=window_size),
            default_conv(n_feats, n_feats, 3),
            AMESA(in_channels)
        ]
        return nn.Sequential(*m_core_list)

    def ablation_1(self, in_channels: int, n_feats: int, depth: int, window_size: int):
        """测试Transformer的效果和卷积的对比

        Args:
            in_channels (int): input channels
            n_feats (int): feature channels
            depth (int): _description_
            window_size (int): _description_

        Returns:
            _type_: _description_
        """
        m_core_list = [
            AMESA(in_channels),
            default_conv(n_feats, n_feats, 3),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, 3),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, 3),
            AMESA(in_channels)
        ]
        return nn.Sequential(*m_core_list)

    def ablation_2(self, in_channels: int, n_feats: int, depth: int, window_size: int):
        """测试ESA和SA的对比

        Args:
            in_channels (int): input channels
            n_feats (int): feature channels
            depth (int): _description_
            window_size (int): _description_

        Returns:
            _type_: _description_
        """
        m_core_list = [
            ESA(in_channels),
            SwinTransformer(n_feats=n_feats,
                            depth=depth,
                            window_size=window_size),
            default_conv(n_feats, n_feats, 3),
            ESA(in_channels),
        ]
        return nn.Sequential(*m_core_list)
    

class HBCT(nn.Module):
    def __init__(self, in_channels: int, n_feats: int, depth: int = 2, window_size: int = 8, num_modules: int = 4) -> None:
        super().__init__()
        m_core_list = [self.create_ctb(
            in_channels, n_feats, depth, window_size) for _ in range(num_modules)]
        # m_core_list = [self.ablation_2(in_channels, n_feats, depth, window_size) for _ in range(
        #     num_modules)]  # Transformer vs Conv
        self.fusion = nn.Sequential(default_conv(
            n_feats * num_modules, n_feats, 1))
        self.core = nn.ModuleList(m_core_list)
        self.num_modules = num_modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = []
        for i in range(self.num_modules):
            x = self.core[i](x)
            local_features.append(x)
        out_fused = self.fusion(torch.cat(local_features, 1))
        return out_fused

    def create_ctb(self, in_channels: int, n_feats: int, depth: int, window_size: int):
        """这里需要测试ESA的重要性(去掉)

        Args:
            in_channels (int): _description_
            n_feats (int): _description_
            depth (int): _description_
            window_size (int): _description_

        Returns:
            _type_: _description_
        """
        m_core_list = [
            ESA(in_channels),
            SwinTransformer(n_feats=n_feats,
                            depth=depth,
                            window_size=window_size),
            default_conv(n_feats, n_feats, 3),
            ESA(in_channels)
        ]
        return nn.Sequential(*m_core_list)

    def ablation_1(self, in_channels: int, n_feats: int, depth: int, window_size: int):
        """测试Transformer的效果和卷积的对比

        Args:
            in_channels (int): input channels
            n_feats (int): feature channels
            depth (int): _description_
            window_size (int): _description_

        Returns:
            _type_: _description_
        """
        m_core_list = [
            ESA(in_channels),
            default_conv(n_feats, n_feats, 3),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, 3),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, 3),
            ESA(in_channels)
        ]
        return nn.Sequential(*m_core_list)

    def ablation_2(self, in_channels: int, n_feats: int, depth: int, window_size: int):
        """测试ESA和SA的对比

        Args:
            in_channels (int): input channels
            n_feats (int): feature channels
            depth (int): _description_
            window_size (int): _description_

        Returns:
            _type_: _description_
        """
        m_core_list = [
            Spatial_Attention(3),
            SwinTransformer(n_feats=n_feats,
                            depth=depth,
                            window_size=window_size),
            default_conv(n_feats, n_feats, 3),
            ESA(in_channels),

            #    Spatial_Attention(3)
        ]
        return nn.Sequential(*m_core_list)


class HCT(nn.Module):
    def __init__(self, scale: int = 4, in_channels: int = 1, n_feats: int = 64, num_modules: int = 4, out_channels=1, num_feats=2):
        super(HCT, self).__init__()
        m_head = [default_conv(in_channels, n_feats, 3)]
        m_body = HBCT(n_feats, n_feats)
        m_tail = [UpSampler(scale, n_feats), default_conv(
            n_feats, out_channels, 3)]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(m_body)
        self.tail = nn.Sequential(*m_tail)
        self.num_modules = num_modules

    def forward(self, input, feats):
        input = self.head(input)
        out_fused = self.body(input) + input
        output = self.tail(out_fused)
        return output


class HCTSF(nn.Module):
    """HCT of sar feature
    """

    def __init__(self, scale: int = 4, in_channels: int = 1, n_feats: int = 64, num_modules: int = 4, out_channels=1, num_feats=4) -> None:
        super(HCTSF, self).__init__()
        m_head = [default_conv(in_channels, n_feats, 3)]
        feats_trans = nn.ModuleList(
            [nn.Conv2d(num_feats, n_feats, 3, 2, 1),
             nn.Sequential(nn.ReLU(inplace=True),
             nn.Conv2d(n_feats, n_feats, 3, 2, 1)),
             ])
        m_body = [HBCT(n_feats, n_feats)]
        m_tail = [
            nn.Sequential(
                default_conv(n_feats, 4 * n_feats, 3, bias=True),
                nn.PixelShuffle(2)
            ),
            nn.Sequential(
                default_conv(n_feats, 4 * n_feats, 3, bias=True),
                nn.PixelShuffle(2)
            ),
            default_conv(n_feats, out_channels, 3),
        ]
        self.head = nn.Sequential(*m_head)
        self.feats_trans = feats_trans
        self.low_fuse = nn.Sequential(default_conv(
            n_feats*2, n_feats, 3), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*m_body)
        self.tail = nn.ModuleList(m_tail)
        self.num_modules = num_modules

    def forward(self, input, feats):
        input = self.head(input)
        feats = torch.cat(feats, 1)
        feat0 = self.feats_trans[0](feats)  # 64, 96, 96
        feat1 = self.feats_trans[1](feat0)  # 64, 48, 48
        out_fused = self.low_fuse(torch.cat([input, feat1], 1))  # exp_14
        # out_fused = input
        out_fused = self.body(out_fused) + out_fused
        # SR
        output_0 = self.tail[0](out_fused)  # 48 -> 96 exp_14
        output_1 = self.tail[1](output_0)  # 96 -> 192 exp_14
        # 48 -> 96 exp_15 exp_16 exp_17
        # output_0 = self.tail[0](out_fused * feat1)
        # 96 -> 192 exp_15 exp_16 exp_17
        # output_1 = self.tail[1](output_0 * feat0)
        output = self.tail[2](output_1)  # channels: 64->1
        return output

    # exp_14
    # 记录的是在body前对feats进行融合 work
    # exp_15
    # 记录的是在body后对feats进行逐分辨率融合，融合方式为拼接 not work
    # exp_16
    # 记录的是在body后对feats进行逐分辨率融合,融合方式为直接相加 not work
    # exp_17
    # 保留exp_14, 在body后对feats进行逐分辨率融合,融合方式为直接相乘 not work

class TC(nn.Module):
    """HCT of sar feature
    """

    def __init__(self, scale: int = 4, in_channels: int = 1, n_feats: int = 64, num_modules: int = 4, out_channels=1, num_feats=4) -> None:
        super(TC, self).__init__()
        m_head = [default_conv(in_channels, n_feats, 3)]
        feats_trans = nn.ModuleList(
            [nn.Conv2d(num_feats, n_feats, 3, 2, 1),
             nn.Sequential(nn.ReLU(inplace=True),
             nn.Conv2d(n_feats, n_feats, 3, 2, 1)),
             ])
        m_body = [ATCB(n_feats, n_feats)]
        m_tail = [
            nn.Sequential(
                default_conv(n_feats, 4 * n_feats, 3, bias=True),
                nn.PixelShuffle(2)
            ),
            nn.Sequential(
                default_conv(n_feats, 4 * n_feats, 3, bias=True),
                nn.PixelShuffle(2)
            ),
            default_conv(n_feats, out_channels, 3),
        ]
        self.head = nn.Sequential(*m_head)
        self.feats_trans = feats_trans
        self.low_fuse = nn.Sequential(default_conv(
            n_feats*2, n_feats, 3), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*m_body)
        self.tail = nn.ModuleList(m_tail)
        self.num_modules = num_modules

    def forward(self, input, feats):
        input = self.head(input)

        #加特征时
        feats = torch.cat(feats, 1)
        feat0 = self.feats_trans[0](feats)  # 64, 96, 96
        feat1 = self.feats_trans[1](feat0)  # 64, 48, 48
        out_fused = self.low_fuse(torch.cat([input, feat1], 1))  # exp_14
        # out_fused = input
        out_fused = self.body(out_fused) + out_fused
        # SR
        output_0 = self.tail[0](out_fused)  # 48 -> 96 exp_14
        output_1 = self.tail[1](output_0)  # 96 -> 192 exp_14
        # 48 -> 96 exp_15 exp_16 exp_17
        # output_0 = self.tail[0](out_fused * feat1)
        # 96 -> 192 exp_15 exp_16 exp_17
        # output_1 = self.tail[1](output_0 * feat0)
        x = self.tail[2](output_1)  # channels: 64->1 

        return x

    # exp_14
    # 记录的是在body前对feats进行融合 work
    # exp_15
    # 记录的是在body后对feats进行逐分辨率融合，融合方式为拼接 not work
    # exp_16
    # 记录的是在body后对feats进行逐分辨率融合,融合方式为直接相加 not work
    # exp_17
    # 保留exp_14, 在body后对feats进行逐分辨率融合,融合方式为直接相乘 not work