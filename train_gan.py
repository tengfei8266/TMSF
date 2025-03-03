# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/06/10 11:22:40
@Author  :   ztf 
@Contact :   tengfeizhang@whu.edu.cn
@License :   (C)Copyright 2024, Wuhan University
@Desc    :   None
'''

import time
import datetime
from src.data import create_dataset
from src.data.prefetcher import DataPrefetcher
from src.utils.saver import Saver
from src.utils.summaries import TensorboardSummary
import logging
from src.model import create_model
from src.option import args
from src.utils.set_seed import setup_seed

setup_seed(args.seed)


if __name__ == '__main__':
    train_dataset_path = args.dataset_dir + "train_" + str(args.scale) +"x.txt"
    dataloader, num = create_dataset(args, train_dataset_path, mode='train')
    model = create_model(args=args)
    saver = Saver(args)
    saver.save_experiment_config()
    summary = TensorboardSummary(saver.experiment_dir)
    logging.basicConfig()
    writer = summary.create_summart()
    learning_rate = args.lr
    iters_per = len(dataloader) 
    for epoch in range(args.epochs):
        train_start = time.time()
        train_prefetcher = DataPrefetcher(dataloader)
        hr,lr = train_prefetcher.next()
        iters = 1
        train_loss_G, train_loss_D  = 0.0, 0.0
        while hr is not None:
            model.set_input(lr, hr)
            model.optimize_parameters()
            train_loss_G += model.loss_G.item()
            train_loss_D += model.loss_D.item()  # type:ignore

            global_step = iters + iters_per * epoch
            if global_step % 20 == 0:
                writer.add_scalar("train/train_loss_D", train_loss_D / iters, global_step)
                writer.add_scalar("train/train_loss_G", train_loss_G/iters, global_step)
                msg = "%s | Epoch: %d | global_step: %d | lr: %.8f | Train loss_D_average: %.4f | Train loss_G_average: %.4f " % (
                    datetime.datetime.now(), epoch, global_step, learning_rate, train_loss_D/iters, train_loss_G/iters)
                print(msg)
                logging.info(msg)
            iters += 1
            hr,lr = train_prefetcher.next()
        writer.add_scalar("train/learning_rate",
                               learning_rate, epoch)
        writer.add_scalar(
            "train/loss_epoch_D", train_loss_D/(iters+1), epoch)
        writer.add_scalar(
            "train/loss_epoch_G", train_loss_G/(iters+1), epoch)
        # 计算参数量
        print("Train:")
        print('[Epoch: %d, numImages: %5d]' % (epoch, num))
        print('Loss of Discriminator: %.3f    Loss of Generator:%.3f' % (train_loss_D/iters, train_loss_G/iters))
        params_num = model.get_params_num()
        print("Params: %.2fM" % (params_num / 1e6))
        logging.info("Train:")
        logging.info('[Epoch: %d, numImages: %5d]' %
                     (epoch, num))
        logging.info('Loss of Discriminator: %.3f    Loss of Generator:%.3f' % (train_loss_D/iters, train_loss_G/iters))
        logging.info("Params: %.2fM" % (params_num / 1e6))
        learning_rate = model.update_learning_rate()

        train_end = time.time()
        print("train time: ", train_end - train_start)
        logging.info("train time: ", train_end - train_start)
        if epoch % 5 == 0:
            model.save_networks(epoch)


