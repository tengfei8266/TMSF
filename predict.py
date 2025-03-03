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
os.environ['PROJ_LIB'] = '/home/ztf/anaconda3/envs/ztf/share/proj'
import re
from collections import OrderedDict
import numpy as np
import torch
from osgeo import gdal
from src.option import args
from src.model.modules import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gdal.DontUseExceptions()
# def to_tensor


def to_tensor(img):
    """convert numpy array to tensor

    Args:
        img (_type_): _description_

    Returns:
        tensor: tensor
    """
    img_new = img[:, :, np.newaxis].astype(np.float32)
    img_new = torch.from_numpy(img_new)
    img_new = img_new.permute(2, 0, 1).unsqueeze(0)
    return img_new


def load_data(lr_path: str):
    # lr
    lr_dataset = gdal.Open(lr_path)
    lr = lr_dataset.ReadAsArray()
    # to Tensor
    lr = to_tensor(lr)
    # hr
    hr_path = re.sub(r'/LR/', r'/HR/',
                     re.sub(r'_(\d+)_(\d+)_', r'_192_192_', lr_path))
    hr_dataset = gdal.Open(hr_path)
    hr = hr_dataset.ReadAsArray()
    # to Tensor
    hr = to_tensor(hr)
    # save proj
    proj = hr_dataset.GetProjection()
    transf = hr_dataset.GetGeoTransform()
    # add feats
    Aspect_path = re.sub(r'/HR/', r'/Feats/Aspect/', hr_path)
    cc_path = re.sub(r'/HR/', r'/Feats/cc/', hr_path)
    mli_path = re.sub(r'/HR/', r'/Feats/mli/', hr_path)
    Slope_path = re.sub(r'/HR/', r'/Feats/Slope/', hr_path)

    Aspect =  gdal.Open(Aspect_path).ReadAsArray() 
    cc =  gdal.Open(cc_path).ReadAsArray()
    mli = gdal.Open(mli_path).ReadAsArray()
    Slope =  gdal.Open(Slope_path).ReadAsArray()
    
    # toTensor   [Aspect,cc, mli,Slope]
    feats = [to_tensor(i).cuda() for i in [Aspect,cc, mli,Slope]]
    return lr.cuda(), hr.cuda(), feats, proj, transf


def load_model(model_name: str, cpt_path: str | None = None, scale: int = 4):
    model = globals()[model_name]
    model = model(scale).cuda()
    if cpt_path is not None:
        checkpoint = torch.load(cpt_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k,v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    # model = create_model(args=args)
    model.eval()
    return model


def merge_dem(input_dir: str, output_dir: str):
    pass


if __name__ == '__main__':
    model_name = args.model_name
    data_path = "./Data/SAR/LR"
    model = load_model(
        model_name, args.resume_SR, 4)  # TODO
    driver = gdal.GetDriverByName('GTiff')

    # 第一张图
    data_list = sorted([name for name in os.listdir(
        data_path) if name.endswith(".tif")])
    # 两张图（研究区域A和B）
    list_length = len(data_list)
    end_of_list_1 = int(list_length * 1 / 2)
    start_of_list_2 = end_of_list_1
    list_1 = data_list[:end_of_list_1]
    list_2 = data_list[start_of_list_2:]
    
    for index, data in enumerate([list_1, list_2]):
        # save predict dem
        out_dir = f'./predict/{model_name}/{index + 1}'
        if not os.path.exists(os.path.join(out_dir)):
            os.makedirs(os.path.join(out_dir))
        i = 1
        for img in data:
            if not img.endswith(".tif"):
                continue
            img_path = os.path.join(data_path, img)
            lr, hr, feats, proj, transfer = load_data(img_path)
            with torch.no_grad():
                sr = model(lr, feats)
            sr = sr.cpu().numpy()
            hr = hr.cpu().numpy()
            # fill nan
            sr_new = np.where(np.isnan(hr), sr, hr)
            sr_new = np.squeeze(sr_new)
            new_img = driver.Create(os.path.join(out_dir, str(
                i) + ".tif"), 192, 192, 1, gdal.GDT_Float32)
            new_img.SetGeoTransform(transfer)
            new_img.SetProjection(proj)
            new_img.GetRasterBand(1).WriteArray(sr_new)
            del new_img
            i += 1
