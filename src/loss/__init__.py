# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/06/10 11:22:40
@Author  :   ztf 
@Contact :   tengfeizhang@whu.edu.cn
@License :   (C)Copyright 2024, Wuhan University
@Desc    :   None
'''

from __future__ import annotations
import torch
import torch.nn as nn
from src.loss.aspect_loss import AspectLossFunc as AspectLoss
from src.loss.slope_loss import SlopeLossFunc as SlopeLoss


class Loss(object):
    """init loss module
    """

    def __init__(self, weight) -> None:
        self.weight = weight

    #直接超分不加特征时
    def l1_loss(self, sr: torch.Tensor, hr: torch.Tensor, feats = [], epoch: int = -1) -> torch.Tensor:
        # 计算L1损失之前, 进行掩膜处理, 排除掉hr中的无效值
        mask = ~torch.isnan(hr)
        criterion = nn.L1Loss()
        temp_hr = torch.where(mask, hr, sr)

        #TCMF
        Aspect, cc, mli, Slope = torch.zeros((16, 1, 192, 192)).to(sr.device), torch.zeros((16, 1, 192, 192)).to(sr.device), torch.zeros((16, 1, 192, 192)).to(sr.device), torch.zeros((16, 1, 192, 192)).to(sr.device)    

        #TCMF1
        #Aspect, cc, mli, Slope = torch.zeros((16, 1, 192, 192)).to(sr.device), torch.zeros((16, 1, 192, 192)).to(sr.device), torch.zeros((16, 1, 192, 192)).to(sr.device), torch.zeros((16, 1, 192, 192)).to(sr.device)
        
        # # TODO: 是否考虑空值区域的损失,取平均值时有差异

        #直接超分不加特征时去掉
        if len(feats):
        # 对应的特征约束(cc: 相干系数; mli: 强度图; Slope坡度, Aspect坡向)
 
            #TCMF
            Aspect,cc, mli, Slope= feats 

            #TCMF1
            Aspect,cc, mli, Slope= feats 

        #传统模型

        """loss = criterion(sr[mask], hr[mask])"""

        """loss = criterion(sr[mask], hr[mask])+ \
        self.SlopeLoss(sr, temp_hr, 3)+self.AspectLoss(sr, temp_hr, 3)"""

        #TCMF模型
        loss = criterion(sr[mask], hr[mask]) + \
             self.weight[1] * criterion(sr[mask] * cc[mask], hr[mask] * cc[mask]) + \
             self.weight[2] * criterion(sr[mask] * mli[mask], hr[mask] * mli[mask]) + \
             self.SlopeLoss(sr, temp_hr, 3)+self.AspectLoss(sr, temp_hr, 3)
        
       #TCM1模型特征消融

        """loss = criterion(sr[mask], hr[mask])"""

        return loss
    
    

    def tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate tv loss

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: tv_loss
        """
        dh = x[:, :, 1:, :] - x[:, :, :-1, :]
        dw = x[:, :, :, 1:] - x[:, :, :, :-1]
        loss = torch.mean(torch.sum(dh ** 2, dim=(2, 3)) +
                          torch.sum(dw ** 2, dim=(2, 3)))
        return loss

    def l2_loss(self, x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        criterion = nn.MSELoss(reduction=reduction)
        loss = criterion(x, y)
        return loss

    def SlopeLoss(self, sr, hr, resolution):
        criterion = SlopeLoss(epsilon=1e-8, resolution=resolution)
        loss = criterion(sr, hr)
        return loss
    
    def AspectLoss(self, sr, hr, resolution):
        criterion = AspectLoss(epsilon=1e-8, resolution=resolution)
        loss = criterion(sr, hr)
        return loss