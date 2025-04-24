
import logging
from .base_model import BaseModel
import models.networks as networks
from models.modules.Basic import Quantization

logger = logging.getLogger('base')


class HCDModel(BaseModel):
    def __init__(self, opt):
        super(HCDModel, self).__init__(opt)

        self.opt = opt
        self.net_Downsample = networks.define_Downsample(opt).to(self.device)
        self.netG = networks.define_G(opt).to(self.device)
        self.Quantization = Quantization()

        self.sr = None

        self.load()

    def load(self):
        if self.opt.get('path') is not None:
            load_path_G = self.opt['path']['pretrain_model_G']
            if load_path_G is not None:
                logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

            load_path_Downsample = self.opt['path']['pretrain_model_Downsample']
            if load_path_Downsample is not None:
                logger.info('Loading model for Downsample [{:s}] ...'.format(load_path_Downsample))
                self.load_network(load_path_Downsample, self.net_Downsample, self.opt['path']['strict_load'])

    def freeze(self):
        for param in self.netG.parameters():
            param.requires_grad = False
        for param in self.net_Downsample.parameters():
            param.requires_grad = False

    def get_downsample(self, HR):
        self.net_Downsample.eval()
        lr = self.net_Downsample(HR)
        lr_quant = self.Quantization(lr)

        return  lr_quant

    def get_upsample(self, LR):
        self.netG.eval()
        sr = self.netG(LR)

        return sr

    def my_test(self, LR):
        self.netG.eval()

        sr = self.netG(LR)
        self.sr = sr

        lr = self.get_downsample(sr)

        return sr, lr