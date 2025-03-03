# -*- encoding: utf-8 -*-
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
scale = [3]
SCALE_OF_TRAIN = 0.8

base_dir = "/home/cgd/DEM/UnsupervisedSR/Data/"

dataset_dir = "Dataset/"

for scale in scale:
    hr_dir = base_dir + "Dataset_LP/" + 'HR'
    lists_hr = sorted([name for name in os.listdir(hr_dir) if name.endswith(".TIF") ])

    lr_dir = base_dir + dataset_dir + 'LR/x' + str(scale) + '/Dem'
    lists_lr = sorted([name for name in os.listdir(lr_dir) if name.endswith(".TIF") and ("patch1" in name or "patch2" in name) ])
    val_hr_dir = base_dir + dataset_dir + 'HR/x' + str(scale)

    lists_lr_train = random.sample(lists_lr, int(len(lists_lr) * SCALE_OF_TRAIN))
    lists_lr_val = [i for i in lists_lr if i not in lists_lr_train]
    lists_hr_val = [ i.replace(str(192//scale) + '_' + str(192//scale), "192_192") for i in lists_lr_val]
    train_list = [os.path.join(hr_dir, i) + " " + \
     os.path.join(lr_dir, j) + "\n" for i,j in zip(lists_hr, lists_lr_train)]
    val_list =  [os.path.join(val_hr_dir, i) + " " + \
     os.path.join(lr_dir, j) + "\n" for i,j in zip(lists_hr_val, lists_lr_val)]

    print("num of Train_dataset:", len(train_list))
    print("num of Val_dataset:", len(val_list))

    with open(base_dir + dataset_dir + "train_" + str(scale) + "x.txt", "w") as f:
        for name in train_list:
            f.write(name)

    with open(base_dir + dataset_dir +  "val_" + str(scale) + "x.txt", "w") as f:
        for name in val_list:
            f.write(name)
