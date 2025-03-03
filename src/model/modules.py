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
import torch.nn.functional as F
from src.model.common import *
from src.model.HST import HST
from src.model.HCT import TC, HCT, HCTSF
from src.model.DAT import DAT


class EDSR(nn.Module):
    """A module packaging EDSR
    """

    def __init__(self, scale: int, mean: float = 0., std: float = 0.) -> None:
        """_summary_

        Args:
            scale (int): scale of reconstruction.
        """
        super(EDSR, self).__init__()
        m_head = [default_conv(1, 256, 3)]
        m_body = [ResBlock(256, 3) for _ in range(32)]
        m_body.append(default_conv(256, 256, 3))  # type: ignore
        m_tail = [UpSampler(scale, 256)]
        m_tail.append(default_conv(256, 1, 3))  # type: ignore
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, lr: torch.Tensor, feats) -> torch.Tensor:
        x = self.head(lr)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


class RCAN(nn.Module):
    """A module packaging RCAN
    """

    def __init__(self, scale: int, mean: float = 0., std: float = 0.) -> None:
        """_summary_

        Args:
            scale (int): scale of reconstruction.

        """
        super(RCAN, self).__init__()
        m_head = [default_conv(1, 64, 3)]
        m_body = [ResidualGroup(64, 3, 16, 20) for _ in range(10)]
        m_body.append(default_conv(64, 64, 3))  # type: ignore
        m_tail = [UpSampler(scale, 64)]
        m_tail.append(default_conv(64, 1, 3))  # type: ignore
        # mean std
        # self.sub_mean = MeanShift(mean, std)
        # self.add_mean = UnNormalize(mean, std)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, lr: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        # x = self.sub_mean(lr)
        x = self.head(lr)
        res = self.body(x)
        res += x
        results = self.tail(res)
        # results = self.add_mean(results)
        return results


class SRCNN(nn.Module):
    def __init__(self, scale: int, mean: float = 0., std: float = 0.) -> None:
        super(SRCNN, self).__init__()
        act = nn.ReLU(True)
        m_body = [default_conv(1, 64, 3), act, default_conv(
            64, 32, 1), act, default_conv(32, 1, 5)]
        self.scale = scale
        self.body = nn.Sequential(*m_body)

    def forward(self, lr: torch.Tensor, feats) -> torch.Tensor:
        return self.body(F.interpolate(lr.float(), scale_factor=self.scale, mode="bicubic"))


class RFAN(nn.Module):
    def __init__(self, scale: int, mean: float = 0., std: float = 0.) -> None:
        super().__init__()
        m_head = [default_conv(1, 64, 3)]
        
        feats_trans = nn.ModuleList(
            [nn.Conv2d(4, 64, 3, 2, 1),
             nn.Sequential(nn.ReLU(inplace=True),
             nn.Conv2d(64, 64, 3, 2, 1)),
             ])
        m_body = [RFA(64) for _ in range(30)]
        m_body.append(default_conv(64, 64, 3))  # type: ignore
        m_tail = [UpSampler1(scale, 64)]
        m_tail.append(default_conv(64, 1, 3))  # type: ignore
        self.head = nn.Sequential(*m_head)
        self.feats_trans = feats_trans
        self.low_fuse = nn.Sequential(default_conv(
            64*2, 64, 3), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, lr: torch.Tensor, feats) -> torch.Tensor:
        x = self.head(lr)
        #加入特征
        feats = torch.cat(feats, 1)
        feat0 = self.feats_trans[0](feats)  # 64, 96, 96
        feat1 = self.feats_trans[1](feat0)  # 64, 48, 48
        out_fused = self.low_fuse(torch.cat([x, feat1], 1))
        res = self.body(out_fused)
        res += x

        res = self.body(x)
        res += x

        results = self.tail(res)
        return results

class BICUBIC(nn.Module):
    def __init__(self, scale: int, mean: float = 0., std: float = 0.) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, lr: torch.Tensor, feats: list[torch.Tensor]):
        return F.interpolate(lr.float(), scale_factor=self.scale, mode="bicubic")


class RDN(nn.Module):
    def __init__(self, scale: int, mean: float = 0., std: float = 0.) -> None:
        super().__init__()
        G0, G1, D, C = 64, 64, 30, 8
        # head
        self.sfe1 = default_conv(1, 64, 3)
        self.sfe2 = default_conv(64, 64, 3)
        # Residual dense blocks
        self.rdbs = nn.ModuleList([RDB(G0, G1, C) for _ in range(D)])
        self.gff = nn.Sequential(default_conv(G1*D, G0, 1),
                                 default_conv(G0, G0, 3))
        self.tail = nn.Sequential(UpSampler(scale, 64), default_conv(64, 1, 3))

    def forward(self, lr: torch.Tensor, feats) -> torch.Tensor:
        sfe1 = self.sfe1(lr)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(30):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        return self.tail(x)



class ARBRCAN(nn.Module):
    """A module packaging ArbRCAN
    """

    def __init__(self, scale: int, mean: float = 0., std: float = 0.) -> None:
        """_summary_

        Args:
            scale (int): scale of reconstruction.

        """
        super(ARBRCAN, self).__init__()
        m_head = [default_conv(1, 64, 3)]
        m_body = [ResidualGroup(64, 3, 16, 20) for _ in range(10)]
        m_body.append(default_conv(64, 64, 3))  # type: ignore
        # m_tail = []
        m_tail = [UpSampler(2, 64)]
        m_tail.append(default_conv(64, 1, 3))  # type: ignore
        sa_adapt = [SA_Adapt(64) for _ in range(10)]
        # mean std
        self.sub_mean = MeanShift(mean, std)
        self.add_mean = UnNormalize(mean, std)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.sa_adapt = nn.Sequential(*sa_adapt)
        # scale-aware upsampling layer
        self.sa_upsample = SA_upsample(scale, 64)
        self.scale = scale

    def forward(self, lr: torch.Tensor, feats) -> torch.Tensor:
        x = self.sub_mean(lr)
        x = self.head(x)
        res = x
        for i in range(10):
            res = self.body[i](res)
            # res = self.sa_adapt[i](res, self.scale)
        res = self.body[-1](res)
        res += x
        # res = self.sa_upsample(res)
        results = self.tail(res)
        results = F.interpolate(
            results.float(), scale_factor=2, mode='bicubic')
        results = self.add_mean(results)
        return results


class ARBEDSR(nn.Module):
    """A module packaging ArbEDSR
    """

    def __init__(self, scale: int, mean: float, std: float) -> None:
        """_summary_

        Args:
            scale (int): scale of reconstruction.

        """
        super(ARBEDSR, self).__init__()
        m_head = [default_conv(1, 64, 3)]
        m_body = [nResBlock(64, 64, 3) for _ in range(32)]
        m_body.append(default_conv(64, 64, 3))   # type: ignore
        # m_tail = []
        m_tail = []
        m_tail.append(default_conv(64, 1, 3))  # type: ignore
        sa_adapt = [SA_Adapt(64) for _ in range(10)]
        # mean std
        self.sub_mean = MeanShift(mean, std)
        self.add_mean = UnNormalize(mean, std)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.sa_adapt = nn.Sequential(*sa_adapt)
        # scale-aware upsampling layer
        self.sa_upsample = SA_upsample(scale, 64)
        self.scale = scale

    def forward(self, lr: torch.Tensor, feats) -> torch.Tensor:
        x = self.sub_mean(lr)
        x = self.head(x)
        res = x
        for i in range(32):
            res = self.body[i](res)
            if i % 4 == 0:
                res = self.sa_adapt[i//4](res, self.scale)
            # res = self.sa_adapt[i](res, self.scale)
        res = self.body[-1](res)
        res += x
        res = self.sa_upsample(res)
        results = self.tail(res)
        # results = F.interpolate(results.float(), scale_factor=1.25, mode='bicubic')
        results = self.add_mean(results)
        return results


class HNCT(nn.Module):
    def __init__(self, scale: int = 4, mean: float = 0., std: float = 0., in_channels: int = 1, n_feats: int = 64, num_modules: int = 4, out_channels=1, num_feats: int = 4):
        super(HNCT, self).__init__()
        self.model = HCTSF(scale=scale, in_channels=in_channels, n_feats=n_feats,
                         num_modules=num_modules, out_channels=out_channels, num_feats=num_feats)

    def forward(self, input, feats):
        # 维度对齐
        return self.model(input, feats)


class HSTN(nn.Module):
    def __init__(self, scale: int = 4, mean: float = 0., std: float = 0., image_size: int = 48):
        super(HSTN, self).__init__()
        self.scale = scale
        self.model = HST(img_size=image_size)

    def forward(self, x, feats):
        return self.model(x)


class DATN(nn.Module):
    def __init__(self, scale: int = 4, mean: float = 0., std: float = 0., image_size: int = 48):
        super(DATN, self).__init__()
        self.scale = scale
        self.model = DAT(img_size=image_size, in_chans=1, num_heads=[
                         6, 6, 6, 6, 6, 6], split_size=[8, 16], upscale=4, depth=[6, 6, 6, 6, 6, 6], expansion_factor=2)

    def forward(self, x, feats):
        return self.model(x)
    
 

class TMSF(nn.Module):
    def __init__(self, scale: int = 4, mean: float = 0., std: float = 0., in_channels: int = 1, n_feats: int = 64, num_modules: int = 4, out_channels=1, num_feats: int = 4):
        super(TMSF, self).__init__()
        self.model = TC(scale=scale, in_channels=in_channels, n_feats=n_feats,
                         num_modules=num_modules, out_channels=out_channels, num_feats=num_feats)

    def forward(self, input, feats):
        # 维度对齐
        return self.model(input, feats)




if __name__ == "__main__":
    model = RFAN(scale=4)
    x = torch.randn(16, 1, 48, 48)
    y = model(x)
    print(y.shape)
