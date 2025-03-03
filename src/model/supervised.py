import os
import torch
from torch.nn.parallel import DistributedDataParallel
import itertools
from .base_model import BaseModel
from src.model.modules import *
from src.loss import Loss


class Supervised(BaseModel):
    def __init__(self, args) -> None:
        BaseModel.__init__(self, args)
        self.args = args
        model = globals()[args.model_name]

        #分布式训练时 #if torch.cuda.device_count() > 1:  分布式训练出现bug时可以添加 find_unused_parameters=True
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", int(local_rank))
        model = DistributedDataParallel(model(
            scale=args.scale, mean=args.mean, std=args.std).to(device), device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
        
        # Launch训练时:
        """model = model(scale=args.scale, mean=args.mean,
            std=args.std).cuda() # Bicubic验证集时可取消这里的注释,其他时候默认即可"""
        
        setattr(self, args.model_name, model)
        self.model_names = [args.model_name]
        self.initialization(args)

    def set_input(self, lr, hr, feats):
        self.hr = hr
        self.lr = lr

        #传统模型
        #self.feats =[feats[i] for i in [0,1,2,3]]

        #TCMF模型
        #self.feats =[feats[i] for i in [0,1,2,3]]


        #TCMF特征消融
        self.feats =[feats[i] for i in [0,1,2,3]]


    def forward(self, mode="train") -> None:
        # 当要增加新的输入，对模型进行约束的时候，可以在这里将self.feats进行取出(self.lr,self.feats)
        #加特征时
        self.output = getattr(self, self.args.model_name)(self.lr,self.feats)


    def initialization(self, args):
        super().initialization()
        if args.isTrain:
            self.optimizers.append(torch.optim.Adam(itertools.chain(
                getattr(self, self.args.model_name).parameters()), lr=args.lr, betas=[args.beta1, args.beta2], eps=args.epsilon))  # type: ignore
            self.loss = Loss(args.weight)
            self.schedulers = [self.get_scheduler(
                optimizer)for optimizer in self.optimizers]
        elif args.resume_SR:
            self.load_networks(
                getattr(self, self.args.model_name), args.resume_SR)

    def optimize_parameters(self, epoch) -> None:
        self.optimizers[0].zero_grad()
        self.forward()
        self.backward(epoch)
        self.optimizers[0].step()

    def backward(self, epoch) -> None:
        loss_total = self.loss.l1_loss(self.output, self.hr, self.feats, epoch)
        self.loss_G = loss_total
        self.loss_D = loss_total
        loss_total.backward()

    def get_scheduler(self, optimizer):
        return super().get_scheduler(optimizer, self.args)
