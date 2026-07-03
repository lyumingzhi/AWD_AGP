from __future__ import print_function, absolute_import

import argparse
import torch
import os
from math import log10
import cv2
import numpy as np

torch.backends.cudnn.benchmark = True

# import official_datasets as datasets
from inpainting.DBWEModel.options import Options
import torch.nn.functional as F
# import pytorch_ssim
# from evaluation import compute_IoU, FScore, AverageMeter, compute_RMSE, normPRED
# from skimage.measure import compare_ssim as ssim
import time
import inpainting.DBWEModel.scripts.models as archs
import torch.nn as nn 
import sys

def is_dic(x):
    return type(x) == type([])

class DBWRnet(nn.Module):
    def __init__(self, args=None):
        super(DBWRnet,self).__init__()
        parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
        sys.argv = ['']
        self.args = parser.parse_args()
        # Machine = models.__dict__[args.models](datasets=data_loaders, args=args)

        # create model
        print("==> creating model ", self.args.arch)
        # self.model = archs.__dict__[self.args.arch]()
        self.model = archs.__dict__['vvv4n']().cuda()
        print("==> creating model [Finish]")

        self.model.cuda()
        

        resume_path = getattr(args, 'dbwe_checkpoint', None) if args is not None else None
        resume_path = resume_path or os.environ.get('AWD_AGP_DBWE_CKPT')
        if not resume_path or not os.path.exists(resume_path):
            raise FileNotFoundError('DBWE checkpoint not found. Set args.dbwe_checkpoint or AWD_AGP_DBWE_CKPT.')
        print("=> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)
        # print('current ckpt', current_checkpoint['arch'])
        # exit()
        if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
            current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module
        
        self.model.load_state_dict(current_checkpoint['state_dict'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, current_checkpoint['epoch']))
        

    def forward(self, x, ):
        # self.model.eval()

        x = x[:,[2,1,0],...]
        imoutput,immask_all,imwatermark = self.model(x)
        # imoutput = imoutput[0] if is_dic(imoutput) else imoutput
        imoutput,imfinal,imwatermark = imoutput[1]*immask_all + x*(1-immask_all),imoutput[0]*immask_all + x*(1-immask_all),imwatermark*immask_all
            
        immask = immask_all[0]
        
        

        # imfinal = imoutput*immask + x*(1-immask)

        # imfinal = (imfinal + 1) / 2

        imoutput = imoutput[:,[2,1,0],:,:]
        imfinal = imfinal[:,[2,1,0],:,:]
        # return G_, g_mask, g_alpha, g_w, I_watermark
        return imoutput, immask, imfinal
    
import cv2
import torchvision.transforms as transforms
