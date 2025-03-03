# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/06/10 11:22:40
@Author  :   ztf 
@Contact :   tengfeizhang@whu.edu.cn
@License :   (C)Copyright 2024, Wuhan University
@Desc    :   None
'''
    
#分步式训练时执行     CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.run --nproc_per_node=3 main.py --model_name "ARBRCAN"
#多模型脚本式训练     bash main.sh 

import os
import time
import torch
import torch.distributed as dist
from src.utils.set_seed import setup_seed
from src.option import args
from src.data import create_dataset
from src.model import create_model
from src.trainer import Trainer

def main() -> None:
    """the entrance of the proj
    """
    setup_seed(args.seed)
    ## 非分布式调用一块显卡，启用launch测试程序
    """os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='9999'
    os.environ['LOCAL_RANK']='1'
    dist.init_process_group(backend="nccl",init_method='env://',rank=0,world_size=1)"""

    ## disributed training分布式训练
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    base_dir = "./Data"
    train_dataset_path = os.path.join(
        base_dir, args.dataset_dir, "train_"+str(int(args.scale))+"x.txt")
    val_dataset_path = os.path.join(
        base_dir, args.dataset_dir, "val_"+str(int(args.scale))+"x.txt")
    train_dataloader, num_train = create_dataset(
        args, train_dataset_path, mode="train")
    val_dataloader, num_val = create_dataset(
        args, val_dataset_path, mode="val")
    loader = {
        "loader_train": train_dataloader,
        "loader_test": val_dataloader,
        "num_train": num_train,
        "num_val": num_val,
    }
    model = create_model(args=args)

    t = Trainer(args, loader, model)
    current_epoch = t.current_epoch
    if args.isTrain:
        for epoch in range(current_epoch, args.epochs):
            train_start = time.time()
            t.train(epoch=epoch)
            val_start = time.time()
            ##分布式训练时
            if local_rank == 0:
               print("train time: ", val_start - train_start)
               t.validation(epoch=epoch)
               print("val time: ", time.time() - val_start)
            ##launch训练时
            """print("train time: ", val_start - train_start)
            t.validation(epoch=epoch)
            print("val time: ", time.time() - val_start)"""
    else:
        t.validation()
    ##分步式训练时
    if local_rank == 0 and args.isTrain:
        t.writer.close()

    ##launch训练时
    """t.writer.close()"""
if __name__ == '__main__':
    main()
