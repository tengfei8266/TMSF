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

import os
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class TensorboardSummary:
    """Tensorboard Summary
    """

    def __init__(self, directory):
        self.directory = directory

    def create_summart(self):
        """create a summaryWriter
        """
        writer = SummaryWriter(logdir=self.directory)
        return writer

    def visualize_image(self, writer, hr, lr, sr, global_step, mode):
        """visualize image

        Args:
            writer (_type_): _description_
            hr (_type_): _description_
            lr (_type_): _description_
            sr (_type_): _description_
            global_step (_type_): _description_
            mode (_type_): _description_
        """
        hr = hr.float()
        hr = hr[:4]
        grid_hr = make_grid(hr, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, 'hr'), grid_hr, global_step)

        lr = lr.float()
        lr = lr[:4]
        grid_lr = make_grid(lr, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, 'lr'), grid_lr, global_step)

        sr = sr.float()
        sr = sr[:4]
        grid_sr = make_grid(sr, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, 'sr'), grid_sr, global_step)

        # slope = slope.float()
        # slope = slope[:4]
        # grid_slope = make_grid(slope, padding=50, normalize=True)
        # writer.add_image(os.path.join(mode, 'slope'), grid_slope, global_step)
