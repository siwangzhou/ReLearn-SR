import os

import torch
import logging
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from archs.DSN_arch import DSN
from .base_model import BaseModel
import utils.util as util
import models.networks as networks
from models.modules.Basic import Quantization

logger = logging.getLogger('base')


class CAR_EDSRModel(BaseModel):

    def __init__(self, opt):
        super(CAR_EDSRModel, self).__init__(opt)

        self.opt = opt

        SCALE = opt['scale']
        KSIZE = 3 * SCALE + 1
        self.scale = SCALE
        self.OFFSET_UNIT = SCALE

        self.net_Downsample = networks.define_Downsample(opt).to(self.device)
        self.netG = networks.define_G(opt).to(self.device)
        self.kernel_generation_net = DSN(k_size=KSIZE, scale=SCALE).to(self.device)


        self.Quantization = Quantization()

        self.load()

    def load(self):
        model_dir = "../experiments/pretrained_models/car_edsr"
        # self.kernel_generation_net.load_state_dict(
        #     torch.load(os.path.join(model_dir, '{0}x'.format(self.scale), 'kgn.pth')), strict=True)

        # self.netG.load_state_dict(torch.load(os.path.join(model_dir, '{0}x'.format(self.scale), 'usn.pth')),
        #                           strict=True)
        # self.net_Downsample.load_state_dict(torch.load(os.path.join(model_dir, '{0}x'.format(self.scale), 'dsn.pth')),
        #                                     strict=True)
        self.load_network(os.path.join(model_dir, '{0}x'.format(self.scale), 'kgn.pth'), self.kernel_generation_net, strict=True)
        self.load_network(os.path.join(model_dir, '{0}x'.format(self.scale), 'usn.pth'), self.netG, strict=True)
        self.load_network(os.path.join(model_dir, '{0}x'.format(self.scale), 'dsn.pth'), self.net_Downsample, strict=True)

    def freeze(self):
        for param in self.netG.parameters():
            param.requires_grad = False
        for param in self.net_Downsample.parameters():
            param.requires_grad = False
        for param in self.kernel_generation_net.parameters():
            param.requires_grad = False

    def get_downsample(self, HR):
        self.kernel_generation_net.eval()
        self.net_Downsample.eval()

        kernels, offsets_h, offsets_v = self.kernel_generation_net(HR)
        lr = self.net_Downsample(HR, kernels, offsets_h, offsets_v, self.OFFSET_UNIT)
        lr_quant = self.Quantization(lr)

        return lr_quant

    def my_upscale(self, LR):
        self.netG.eval()
        SR = self.netG(LR)

        return SR

    def my_test(self, LR):

        self.netG.eval()

        sr = self.netG(LR)
        lr = self.get_downsample(sr)


        return sr, lr
