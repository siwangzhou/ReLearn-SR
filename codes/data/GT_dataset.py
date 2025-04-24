import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import sys
import os

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    from utils import util as utils
except ImportError:
    pass


class GTDataset(data.Dataset):
    '''
    Load GT images only.
    '''

    def __init__(self, opt):
        super(GTDataset, self).__init__()
        self.opt = opt
        self.GT_paths = None
        self.GT_env = None  # environment for lmdb

        if opt['data_type'] == 'img':
            self.GT_paths = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])  # GT list
        else:
            print('Error: data_type is not matched in Dataset')
        assert self.GT_paths, 'Error: GT paths are empty.'

    def __getitem__(self, index):
        resolution = None
        # get GT image
        GT_path = self.GT_paths[index]
        img_GT = util.read_img(self.GT_env, GT_path, resolution)  # return: Numpy float32, HWC, BGR, [0,1]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        return {'GT': img_GT, 'GT_path': GT_path}

    def __len__(self):
        return len(self.GT_paths)


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')
