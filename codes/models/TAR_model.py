import logging
from collections import OrderedDict

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks

from .base_model import BaseModel
from models.modules.Quantization import Quantization

logger = logging.getLogger('base')


class TARModel(BaseModel):
    def __init__(self, opt):
        super(TARModel, self).__init__(opt)

        self.fake_H = None
        self.opt = opt

        self.netG = networks.define_G(opt).to(self.device)


        self.load()

        self.Quantization = Quantization()

    def my_test(self, LR):

        self.netG.eval()

        fake_H = self.my_upscale(LR)

        self.fake_H = fake_H

        fake_L_from_SR = self.get_downsample(fake_H)
        # fake_L_from_SR = self.Quantization(fake_L_from_SR)

        return fake_H, fake_L_from_SR

    def get_downsample(self, HR_img):
        self.netG.eval()

        LR_img = self.netG.encode(HR_img)

        return LR_img

    def my_upscale(self, LR_img, scale=4, gaussian_scale=1):

        self.netG.eval()

        SR_img = self.netG.decode(LR_img)

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
