'''
@File    :   __init__.py
@Time    :   2024/06/10 11:22:40
@Author  :   ztf 
@Contact :   tengfeizhang@whu.edu.cn
@License :   (C)Copyright 2024, Wuhan University
@Desc    :   None
'''
import random

import os

random.seed(27)
scale = "4"

base_dir = "./Data"
# for  dir in os.listdir(base_dir):
# dataset_dir = os.path.join(base_dir, dir)

hr_dir = os.path.join(base_dir, "hr")
lr_dir = os.path.join(base_dir, "lr")

name_lists_hr = sorted([name for name in os.listdir(hr_dir) if name.endswith(".tif") ])
name_lists_lr = sorted([name for name in os.listdir(lr_dir) if name.endswith(".tif") ])

dataset_list = [os.path.join(hr_dir, i) + " " + \
    os.path.join(lr_dir, j) + "\n" for i,j in zip(name_lists_hr, name_lists_lr)]

train_list = random.sample(dataset_list, 2000)
val_list =  [i for i in dataset_list if i not in train_list]
val_list = random.sample(val_list, 330)

with open(base_dir + "/train_" + str(scale) + "x.txt", "w") as f:
    for name in train_list:
        f.write(name)

with open(base_dir +  "/val_" + str(scale) + "x.txt", "w") as f:
    for name in val_list:
        f.write(name)
