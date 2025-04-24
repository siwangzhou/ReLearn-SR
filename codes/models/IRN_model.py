import logging
from collections import OrderedDict

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks

from .base_model import BaseModel
from models.modules.Quantization import Quantization

logger = logging.getLogger('base')


class IRNModel(BaseModel):
    def __init__(self, opt):
        super(IRNModel, self).__init__(opt)

        self.fake_H = None
        self.opt = opt

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.rank = -1  # non dist training
            self.netG = DataParallel(self.netG)

        self.load()

        self.Quantization = Quantization()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def my_feed_data(self, L, H):
        self.ref_L = L.to(self.device)
        self.real_H = H.to(self.device)

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def my_test(self, LR):

        self.netG.eval()

        fake_H = self.my_upscale(LR)

        self.fake_H = fake_H

        fake_L_from_SR = self.get_downsample(fake_H)
        # fake_L_from_SR = self.Quantization(fake_L_from_SR)

        return fake_H, fake_L_from_SR

    def get_downsample(self, HR_img):
        self.netG.eval()
        LR_img = self.netG(x=HR_img)[:, :3, :, :]
        LR_img = self.Quantization(LR_img)

        return LR_img

    def my_upscale(self, LR_img, scale=4, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale ** 2 - 1), Lshape[2], Lshape[3]]
        y_ = torch.cat((LR_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        # self.netG.eval()
        SR_img = self.netG(x=y_, rev=True)[:, :3, :, :]

        return SR_img

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    # 冻结参数
    def freeze(self):
        for param in self.netG.parameters():
            param.requires_grad = False
