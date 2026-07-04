from __future__ import print_function, absolute_import

import argparse
import torch
import os
from inpainting.AWD_AGP.weight_utils import resolve_weight_path
from math import log10
import cv2
import numpy as np

torch.backends.cudnn.benchmark = True

import inpainting.SLBR.src.models as models
from inpainting.SLBR.options import Options
import torch.nn.functional as F
# import pytorch_ssim
# from evaluation import compute_IoU, FScore, AverageMeter, compute_RMSE, normPRED
# from skimage.measure import compare_ssim as ssim
import time
import inpainting.SLBR.src.networks as nets
import torch.nn as nn 
def is_dic(x):
    return type(x) == type([])

class SLBRnet(nn.Module):
    def __init__(self, args=None):
        super(SLBRnet,self).__init__()
        # parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
        # self.args = parser.parse_args()
        # Machine = models.__dict__[args.models](datasets=data_loaders, args=args)
        
        self.args = self.load_config()
        

        self.model = nets.__dict__[self.args.nets](args=self.args)
        self.model.cuda()

        resume_path = resolve_weight_path('weights/SLBR/model_best.pth.tar', args, 'slbr_checkpoint', 'AWD_AGP_SLBR_CKPT', ['inpainting/SLBR/model_best.pth (1).tar'], 'SLBR checkpoint')
        print("=> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)
        if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
            current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module
        
        self.model.load_state_dict(current_checkpoint['state_dict'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, current_checkpoint['epoch']))

    def load_config(self):
        cfg = {'nets': 'slbr', 
                'models': 'slbr',
                'workers': 2,
                'epochs': 30,
                'start-epoch': 0,
                'train-batch': 64,
                'test-batch': 6,
                'lr': 1e-3,
                'dlr': 1e-3,
                'beta1': 0.9,
                'beta2': 0.999,
                'momentum': 0,
                'weight_decay': 0,
                'schedule': [5, 10],
                'gamma': 0.1,
                'flip': False,
                'lambda_l1': 4,
                'lambda_primary': 0.01,
                'lambda_style': 0,
                'lambda_content': 0,
                'lambda_iou': 0,
                'lambda_mask': 1,
                'sltype': 'vggx',
                'alpha': 0.5,
                'sigma-decay': 0,
                'dataset_dir': '',
                'test_dir': '',
                'data': '',
                'checkpoint': '',
                'resume': '',
                'finetune': '',
                'evaluate': True,
                'data-augumentation': False,
                'debug': False,
                'input-size': 256,
                'freq': -1,
                'normalized-input': False, 
                'res': False,
                'requires-grad': False,
                'gpu': True,
                'gpu_id': '0',
                'preprocess': 'resize',
                'crop_size': 256,
                'no_flip': True,
                'masked': False,
                'gan-norm': False,
                'hl': False,
                'loss-type': '12',
                'dataset': 'clwd',
                'name': 'v2',
                'sim_metric': 'cos',
                'k_center': 2,
                'project_mode': 'simple',
                'mask_mode': 'res',
                'bg_mode': 'res_mask',
                'use_refine': True,
                'k_refine': 3,
                'k_skip_stage': 3
                }
        cfg = EasyDict(cfg)
        return cfg
    def forward(self, x, ):
        x = x[:,[2,1,0],...]
        # x = x * 2.0 - 1.0
        imoutput,immask_all,imwatermark = self.model(x)
        imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            
        immask = immask_all[0]

        imfinal = imoutput*immask + x*(1-immask)

        imoutput = imoutput[:,[2,1,0],:,:]
        imfinal = imfinal[:,[2,1,0],:,:]
        # return G_, g_mask, g_alpha, g_w, I_watermark
        return imoutput, immask, imfinal

from typing import Any, List, Tuple, Union

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

import cv2
import torchvision.transforms as transforms
