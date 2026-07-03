import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from inpainting.WDNet.dataloader import dataloader
from inpainting.WDNet.unet_parts import *
from tensorboardX import SummaryWriter
from inpainting.WDNet.vgg import Vgg16

from inpainting.WDNet.WDNet import generator, discriminator

import torchvision

class WDnet(nn.Module):
    def __init__(self, args=None):
        super(WDnet,self).__init__()
        # parameters
        # self.epoch = args.epoch
        # self.batch_size = args.batch_size
        # self.save_dir = args.save_dir
        # self.result_dir = args.result_dir
        # self.dataset = args.dataset
        # self.log_dir = args.log_dir
        # self.gpu_mode = args.gpu_mode
        # self.input_size = args.input_size
        # self.model_name = args.gan_type
        self.z_dim = 62
        self.class_num = 3
        self.sample_num = self.class_num ** 2

        def weight_init(m):
          classname=m.__class__.__name__
          if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data,0.0,0.02)
          elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)
        # networks init
        self.G = generator(3, 3)
        # self.D = discriminator(input_dim=6, output_dim=1)
        # self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        # self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        
        # if self.gpu_mode:
        #     self.G.cuda()
        #     # self.D.cuda()
        #     self.BCE_loss = nn.BCELoss().cuda()
        #     self.l1loss=nn.L1Loss().cuda()
        #     self.loss_mse = nn.MSELoss().cuda()
        # else:
        #     self.BCE_loss = nn.BCELoss()

        self.G.cuda()
        self.G.apply(weight_init)
        # self.D.apply(weight_init)
        #self.load()
        print('---------- Networks architecture -------------')
        #utils.print_network(self.G)
        #utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        # self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        # for i in range(self.class_num):
        #     self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
        #     for j in range(1, self.class_num):
        #         self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        # temp = torch.zeros((self.class_num, 1))
        # for i in range(self.class_num):
        #     temp[i, 0] = i

        # temp_y = torch.zeros((self.sample_num, 1))
        # for i in range(self.class_num):
        #     temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        # self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        # if self.gpu_mode:
        #     self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

        checkpoint = getattr(args, 'wdnet_checkpoint', None) if args is not None else None
        checkpoint = checkpoint or os.environ.get('AWD_AGP_WDNET_CKPT') or 'inpainting/WDNet/WDNet_G.pkl'
        if not os.path.exists(checkpoint):
            raise FileNotFoundError('WDNet checkpoint not found. Set args.wdnet_checkpoint or AWD_AGP_WDNET_CKPT.')
        self.G.load_state_dict(torch.load(checkpoint))
        self.G.eval()
    def forward(self, x, mask):
        x = x[:,[2,1,0],...]
        # x = (x - 1) * 2
        G_ ,g_mask, g_alpha, g_w, I_watermark= self.G(x)
        # G_ = G_ / 2 + 1
        G_ = G_[:,[2,1,0],:,:]
        # return G_, g_mask, g_alpha, g_w, I_watermark
        # return G_, g_mask, x * (1-g_alpha) + g_w * g_alpha
        # print('x', x.shape, 'g_alpha', g_alpha.shape, 'g_w', g_w.shape, 'g_alpha', g_alpha.shape)

        ###########################
        mask = torchvision.transforms.Compose([ torchvision.transforms.Resize((256,256))])(mask)
        # assert cv2.imwrite('WDnet_output_example.png', ((x * (1-g_alpha) + g_w * g_alpha) * mask)[0].detach().cpu().numpy().transpose(1,2,0) * 255)
        return G_, g_mask, (x * (1-g_alpha) + g_w * g_alpha) * mask
        ###############################
        return G_, g_mask, x * (1-g_alpha) + g_w * g_alpha
    
import cv2
import torchvision.transforms as transforms
