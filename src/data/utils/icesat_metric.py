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
import math
import pandas as pd
import numpy as np
from osgeo import gdal
os.environ['PROJ_LIB'] = '/home/cgd/anaconda3/envs/cgd/share/proj'
gdal.DontUseExceptions()


def get_grid_template(lon, lat, transform, dem_arr, grid_size=(21, 21)):
    """获取网格模板

    Args:
        lon (float): 纬度
        lat (float): 经度
        transform (tuple): tif的空间仿射矩阵
        grid_size (tuple, optional): 网格模板大小. Defaults to (21, 21).

    Returns:
        NDArray: 网格内包含的高程坐标
    """
    x_origin, pixel_width, _, y_origin, _, pixel_height = transform
    x_offset = int((lon - x_origin) // pixel_width)
    y_offset = int((lat - y_origin) // pixel_height)

    # 网格 (防止超限)
    y_bottom = int(max(y_offset - grid_size[0]//2, 0))
    x_left = int(max(x_offset - grid_size[1]//2, 0))
    y_top = int(min(y_offset + grid_size[0]//2 + 1, dem_arr.shape[0]))
    x_right = int(min(x_offset + grid_size[1]//2 + 1, dem_arr.shape[1]))
    grid = dem_arr[y_bottom: y_top, x_left: x_right]
    # 展开到一维
    grid = np.array(grid).flatten()
    # 排除空值
    return grid[~np.isnan(grid)]


def cal_metric(grid_template, h):
    """计算指标

    Args:
        grid_template (ndarray): 以点为中心的网格模板，包含高程信息
        h (float): 点的对比高程
    """
    # 计算指标
    center = grid_template[int(len(grid_template)//2)] - h
    h2 = min(grid_template) - h
    h3 = max(grid_template) - h
    mean = np.mean(grid_template) - h
    return center, h2, h3, mean


if __name__ == "__main__":
    # read xlsx
    # 读取这些点
    xlsx_path = "./Data/YA_SAR/ICESat.xlsx"
    df = pd.read_excel(xlsx_path)
    arr = df.values
    # 获取tif的仿射矩阵
    raster_path = "./Data/YA_SAR/Predict/ICHighDEM111.tif"
    dataset = gdal.Open(raster_path)
    transf = dataset.GetGeoTransform()
    dem_arr = dataset.ReadAsArray()
    mean_metric = [0 for _ in range(4)]
    rmse_metric = [0 for _ in range(4)]
    for point in arr:
        lon, lat, h = point
        # 生成格网
        grid_template = get_grid_template(lon, lat, transf, dem_arr)
        cur_metric = cal_metric(grid_template, h)
        mean_metric = [i+j for i, j in zip(mean_metric, cur_metric)]
        rmse_metric = [i+j ** 2 for i, j in zip(mean_metric, cur_metric)]
    # 计算均值和RMSE
    mean = [ i / len(arr) for i in mean_metric]
    rmse = [ math.sqrt(i / len(arr)) for i in rmse_metric]
    print("Mean:", mean)
    print("RMSE:", rmse)
