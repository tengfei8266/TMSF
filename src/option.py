# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/06/10 11:22:40
@Author  :   ztf 
@Contact :   tengfeizhang@whu.edu.cn
@License :   (C)Copyright 2024, Wuhan University
@Desc    :   None
'''

# here put the import lib   用于配置深度学习模型的训练过程

import argparse        #用于构建命令行接口的库

parser = argparse.ArgumentParser(description='unsupervised super-resolution')


# Hardware specifications   硬件规格参数
parser.add_argument('--workers', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--nproc_per_node', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=27,
                    help='random seed')
parser.add_argument('--gpu_ids', type=list,
                    default=[0, 1, 2, 3], help="use which gpu in environ to train")

# Data specifications   数据规格参数

parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--resolution', type=int, default=30,
                    help='the resolution of DEM')
parser.add_argument('--patch_size', type=int, default=100,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=256,
                    help='maximum value of RGB')
parser.add_argument('--n_channels', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--dataset_dir', type=str,
                    default="YA_SAR", help="dataset name")
parser.add_argument('--mean', type=float, default=-2253.41136253128,
                    help='maximum value of RGB')
parser.add_argument('--std', type=float, default=3844.9346,
                    help='maximum value of RGB')

# Model specifications  模型规格参数
parser.add_argument('--model_name', type=str, required=True,
                    help='model name')
parser.add_argument('--model_type', type=str, default="supervised",
                    help='supervised or unsupervised model')
parser.add_argument('--pretrained-path', type=str,
                    default=None)
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_res_blocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_features', type=int, default=256,
                    help='number of feature maps')
parser.add_argument('--isTrain', action="store_false",
                    default=True, help='is test or not')
parser.add_argument('--resume', type=str, default=None, help='checkpoint name')

# DRN model config  DRN代表深度残差网络，这些参数是特定于DRN模型的配置。
parser.add_argument('--n_blocks', type=int, default=40,
                    help="number of DRN blocks")
parser.add_argument('--n_feats', type=int, default=20,
                    help='channels of DRN features ')
parser.add_argument('--eta_min', type=float, default=5e-6,
                    help='eta_min lr')

# pretrained model 预训练模型参数
parser.add_argument('--resume_DT', type=str, default=None,
                    help='domain transfer checkpoint name')
parser.add_argument('--resume_SR', type=str, default=None,
                    help='SR checkpoint name')

# Training specifications 训练规格参数

parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=8,
                    help='input batch size for val')
parser.add_argument('--lr_policy', type=str, default='cosine',
                    help='learning rate policy. [linear | step | plateau | cosine]')


# Optimization specifications 优化规格参数
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--milestones', type=list, default=[300, 450, 600],
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications 损失规格参数
parser.add_argument('--weight', type=list, default=[1, 1, 1],
                    help='loss function weight')

args = parser.parse_args()


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
