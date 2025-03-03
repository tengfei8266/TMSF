# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/06/10 11:22:40
@Author  :   ztf 
@Contact :   tengfeizhang@whu.edu.cn
@License :   (C)Copyright 2024, Wuhan University
@Desc    :   None
'''

import itertools
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from src.model.common import *
from src.loss import Loss
from src.utils.image_pool import ImagePool

Tensor = torch.Tensor


class DomainTransfer(BaseModel):
    def __init__(self, args) -> None:
        BaseModel.__init__(self, args)
        # Generator
        self.args = args
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D1']
            # Discriminator
            self.D1 = Discriminator(in_channels=args.n_channels,
                                    out_channels=args.n_channels, h_stride=1, bias=True).to(0)
        else:  # during test time, only load Gs
            self.model_names = ['G1', 'G2']
        self.G1 = ResGenerator(in_channels=args.n_channels,
                               out_channels=args.n_channels, kernel_size=3, stride=1, bias=True).to(0)
        self.G2 = ResGenerator(in_channels=args.n_channels,
                               out_channels=args.n_channels, kernel_size=3, stride=1, bias=True).to(0)
        self.initialization(args)

    def set_input(self, lr: Tensor, hr: Tensor):
        self.syn_lr = F.interpolate(
            hr.float(), scale_factor=1/self.args.scale, mode="bicubic")
        self.real_lr = lr

    def forward(self):
        # get x of target domain
        self.fake_lr = self.G1(self.syn_lr)  # calculate tv loss
        # self.real_lr = self.G1(self.syn_lr)  # calculate identity loss
        # discriminated results
        # self.dis_fake_lr = self.D1(self.fake_lr)  # calculate adversarial loss
        # transfer x to source domain
        self.rec_syn = self.G2(self.fake_lr)  # calculate cyc loss

    def backward_G(self) -> None:
        """Calculate the loss of generator and update weights
        of generator
        """
        pred_fake = self.D1(self.fake_lr)
        weight = self.args.weight
        loss_gan = self.loss.l2_loss(pred_fake, torch.ones_like(pred_fake))
        loss_tv = self.loss.tv_loss(self.fake_lr)
        loss_cyc = self.loss.l1_loss(self.rec_syn, self.syn_lr)
        loss_idt = self.loss.l2_loss(self.real_lr, self.syn_lr)
        self.loss_G = loss_gan + \
            weight[0] * loss_cyc + weight[1] * loss_idt + weight[2] * loss_tv
        self.loss_G.backward()

    def backward_D(self) -> None:
        """Calculate the loss of discriminator and update weights
        of discriminator
        """
        # real
        pred_real = self.D1(self.real_lr)
        loss_D_real = self.loss.l2_loss(pred_real, torch.ones_like(pred_real))
        # fake
        # fake_lr = self.fake_pool.query(self.fake_lr)
        pred_fake = self.D1(self.fake_lr.detach())
        loss_D_fake = self.loss.l2_loss(pred_fake, torch.zeros_like(pred_fake))
        self.loss_D = loss_D_real + loss_D_fake
        self.loss_D.backward()

    def optimize_parameters(self) -> None:
        """Calculate losses, gradients, and update network weights;
         called in every training iteration"""
        self.forward()
        # Discriminator
        self.set_requires_grad([self.G1, self.G2], False)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # Generate
        self.set_requires_grad([self.G1, self.G2], True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def initialization(self, args) -> None:
        """init optimizers and schedulers
        """
        super(DomainTransfer, self).initialization()
        if self.args.isTrain:
            # self.fake_pool = ImagePool(50)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G1.parameters(),
                                                                self.G2.parameters()), lr=args.lr, betas=[0.5, args.beta2], eps=args.epsilon)  # type: ignore
            self.optimizer_D = torch.optim.Adam(self.D1.parameters(), lr=args.lr, betas=[
                                                0.5, args.beta2], eps=args.epsilon)  # type: ignore
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.loss = Loss(args.weight)
            self.loss_G = 0.0
            self.loss_D = 0.0
            self.schedulers = [self.get_scheduler(
                optimizer, args)for optimizer in self.optimizers]
