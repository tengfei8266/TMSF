from src.model.common import *
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from src.loss import Loss
from .base_model import BaseModel


class CycleCNN(BaseModel):
    def __init__(self, args) -> None:
        BaseModel.__init__(self, args)
        self.args = args
        self.G1 = Cycle_G1(scale=args.scale).cuda()
        self.G2 = Cycle_G2(scale=args.scale).cuda()

        self.model_names = ['G1', "G2"]
        self.initialization(args)

    def set_input(self, lr, hr):
        self.hr = hr
        self.lr = lr

    def forward(self):
        """forward process of the network
        """
        self.output = self.G1(self.lr)
        self.hr_b = F.interpolate(
            self.hr.float(), scale_factor=1/self.args.scale, mode="bicubic")

    def initialization(self, args) -> None:
        """init optimizers and schedulers
        """
        super(CycleCNN, self).initialization()
        params_list = []
        for model in self.model_names:
            net = getattr(self, model)
            params_list.append(net.parameters())
        if args.isTrain:
            self.optimizers.append(torch.optim.Adam(itertools.chain(
                *params_list), lr=args.lr, betas=[args.beta1, args.beta2], eps=args.epsilon))  # type: ignore
            self.loss = Loss(args.weight)
            self.schedulers = [self.get_scheduler(
                optimizer, args)for optimizer in self.optimizers]

    def optimize_parameters(self) -> None:
        """Calculate losses, gradients, and update network weights;
         called in every training iteration
        """
        self.forward()
        self.optimizers[0].zero_grad()
        self.backward()
        self.optimizers[0].step()

    def backward(self):
        """calculate the loss and backward process of the network
        """
        lr_c = self.G2(self.output)
        hr_c = self.G1(self.G2(self.hr))
        loss_cyc = self.loss.l2_loss(
            lr_c, self.lr) + self.loss.l2_loss(hr_c, self.hr)
        loss_idt = self.loss.l2_loss(self.G1(self.hr_b), self.hr)
        self.loss_total = 2*loss_cyc + loss_idt
        self.loss_total.backward()
