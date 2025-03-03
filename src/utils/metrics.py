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

from __future__ import annotations
from torch import Tensor
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

RGB_RANGE = 65536


class Evaluator(object):
    """Evaluation for reconstruction; Metrics containing mse, mae, rmse, e_max,  psnr, slope_mae
    """

    def __init__(self, batch_size: int = 16, rgb_range=RGB_RANGE, scale=3) -> None:
        self.batch_size = batch_size
        self.rgb_range = rgb_range
        self.scale = scale
        self.metric_matrix = [np.zeros((batch_size, 1)) for _ in range(5)]

    def score(self, num: int) -> list[float]:
        """Get score of this dataset

        Args:
            num (int): num of this dataset

        Returns:
            list[float]: Metrics containing mse, mae, rmse, e_max,  psnr, slope_mae
        """
        if num % self.batch_size != 0:
            num_remain = num % self.batch_size
            num -= num_remain
        result = [np.sum(i) / num for i in self.metric_matrix]
        return result

    def __generate_matrix(self, hr: np.ndarray, sr: np.ndarray):
        mask = ~np.isnan(hr)
        temp_hr = np.where(mask, hr, sr)
        diff = sr - temp_hr
        mse = np.mean(diff ** 2, axis=(2, 3))
        # if np.max(mse) > 100:
        #     i = 1
        # psnr = -10 * np.log10(mse/(self.rgb_range**2))
        mae = np.mean((np.abs(diff)), axis=(2, 3))
        rmse = np.sqrt(mse)
        # terrain_num = np.sum(flow == 1, axis = (2,3))
        # terrain_num[terrain_num == 0 ] = 1
        # flow_mae = np.sum((np.abs(diff * flow)), axis=(2,3)) / terrain_num
        e_max = np.max(np.abs(diff), axis=(2, 3))
        slope_mae = self.cac_slope_mae(sr, temp_hr)
        return [mse, mae, rmse, e_max, slope_mae]

    def add_batch(self, sr: Tensor, hr: Tensor):
        """add predict batches

        Args:
            sr (Tensor): _description_
            hr (Tensor): _description_
        """
        sr = sr.data.cpu().numpy()
        hr = hr.data.cpu().numpy()
        # revert normalization
        # sr = ((sr + 1) / 2.0) * (1819 - 1012) + 1012
        # hr = ((hr + 1) / 2.0) * (1819 - 1012) + 1012
        # sr = ((sr + 1) / 2.0) * (593 + 17) - 17
        # hr = ((hr + 1) / 2.0) * (593 + 17) - 17
        assert hr.shape == sr.shape
        # slope_mae = self.cac_slope_mae(sr, hr)
        # diff = hr - sr
        matrix = self.__generate_matrix(hr, sr)  # type: ignore
        # matrix.insert(4, slope_mae)
        self.metric_matrix = [i + j for i,
                              j in zip(self.metric_matrix, matrix)]

    def reset(self):
        """reset metric
        """
        self.metric_matrix = [np.zeros((self.batch_size, 1)) for _ in range(5)]

    def __cac_slope(self, dx: np.ndarray, dy: np.ndarray):
        slope = (np.arctan(np.sqrt(dx ** 2 + dy ** 2))) * 57.295779513
        return slope

    def cac_slope_mae(self, sr, hr):
        sr_offset_x = sr[:, :, :, 2:]
        hr_offset_x = hr[:, :, :, 2:]
        sr_offset_y = sr[:, :, 2:, :]
        hr_offset_y = hr[:, :, 2:, :]
        hr_diff_x = (hr[:, :, :, :-2] - hr_offset_x)[:, :, :-2, :] / self.scale
        sr_diff_x = (sr[:, :, :, :-2] - sr_offset_x)[:, :, :-2, :] / self.scale
        hr_diff_y = (hr[:, :, :-2, :] - hr_offset_y)[:, :, :, :-2] / self.scale
        sr_diff_y = (sr[:, :, :-2, :] - sr_offset_y)[:, :, :, :-2] / self.scale
        hr_slope = self.__cac_slope(hr_diff_x, hr_diff_y)
        sr_slope = self.__cac_slope(sr_diff_x, sr_diff_y)
        slope_mae = np.mean((np.abs(hr_slope - sr_slope)), axis=(2, 3))
        return slope_mae


    # def add_batch(self, hr, sr):
    # sr = sr.astype(np.uint8)
if __name__ == '__main__':
    import pandas as pd
    # plt.rcParams['font.size'] = 20
    mpl.rcParams['font.sans-serif'] = "Times New Roman"
    df = pd.DataFrame(np.random.rand(6, 4), index=[
                      'one', 'two', 'three', 'four', 'five', 'six'], columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
    ax = df.plot(kind='bar', rot=0)
    # plt.rc('font', family='Times New Roman')

    ax.set_ylabel('Weight', rotation=0, x=-0.5)
    ax.set_xlabel('RFM Index')
    plt.show()
    # mse, psnr, mae, rmse, e_max, slope_mae = [random.randint(0,9)  for _ in range(6)] / 5
    # metric_dict = {
    #     "MSE": mse,
    #     "PSNR": psnr,
    #     "MAR": mae,
    # }
    # with open("best_metrics.txt", "w") as f:
    #     f.write(str(metric_dict))
    # mse = mse + 1 + 2 + 3
    # hr =  torch.randn(16,1,192,192)
    c = 1
