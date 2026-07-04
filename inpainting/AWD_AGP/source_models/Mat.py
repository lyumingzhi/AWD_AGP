from pydoc import resolve
from inpainting.MAT.networks.mat import Generator
import torch
import os
from inpainting.AWD_AGP.weight_utils import resolve_weight_path
import torchvision
import inpainting.MAT.dnnlib as dnnlib
import numpy as np
import inpainting.MAT.legacy as legacy

class MatAPI(torch.nn.Module):
    def __init__(self,opt,device='cuda'):
        super(MatAPI,self).__init__()
        self.opt=opt
        self.resolution=512
        self.device=device
        self.module=Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=self.resolution, img_channels=3).to(device).eval().requires_grad_(False)
        #     G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore

        # self.copy_params_and_buffers(G_saved, self.module, require_all=True)

        checkpoint = resolve_weight_path('weights/MAT/pretrained/Places_512_FullData_real.pkl', opt, 'mat_checkpoint', 'AWD_AGP_MAT_CKPT', ['inpainting/MAT/pretrained/Places_512_FullData_real.pkl'], 'MAT checkpoint')
        self.module.load_state_dict(torch.load(checkpoint))
        self.module.eval()
    def forward(self,x,masks,keepFeat=False):
        # masks=1-masks
                
        
        # print('mask', masks)
        # gt_images, masks = self.__cuda__(*items)
        original_x=x.clone()
        x=x[:,[2,1,0],...]
        x=torchvision.transforms.Compose([ torchvision.transforms.Resize((self.resolution,self.resolution))])(x)
        masks=torchvision.transforms.Compose([ torchvision.transforms.Resize((self.resolution,self.resolution))])(masks)

        masks=(masks>0.1).to(self.device)*1.0
        masks=1-masks
        masked_images = x * masks
        # exit()
        # masks = torch.cat([masks]*3, dim = 1)
        # print(masks.size())
        # exit(0)

        masked_images=masked_images*2-1

        label = torch.zeros([1, self.module.c_dim], device=self.device)
        z = torch.from_numpy(np.random.randn(1, self.module.z_dim)).to(self.device)
        truncation_psi=1.0
        noise_mode='const'

        if keepFeat==False:
            fake_B = self.module(masked_images, masks,z,label,truncation_psi=truncation_psi, noise_mode=noise_mode )
        else:
            fake_B, FeatList = self.module(masked_images, masks, z,label,truncation_psi=truncation_psi, noise_mode=noise_mode,keepFeat=keepFeat)

        # if self.opt.wo_mask:
        #     comp_B=fake_B
        # else:
        #     comp_B = fake_B * (   1 - masks) + x * masks
        
        comp_B=fake_B/2+0.5

        # print(torch.max(torch.abs(comp_B-x)))
        # exit()
        comp_B=comp_B[:,[2,1,0],:,:]
        if keepFeat==False:
            return comp_B
        else:
            return comp_B, FeatList
    def copy_params_and_buffers(self,src_module, dst_module, require_all=False):
        assert isinstance(src_module, torch.nn.Module)
        assert isinstance(dst_module, torch.nn.Module)
        src_tensors = {name: tensor for name, tensor in self.named_params_and_buffers(src_module)}
        for name, tensor in self.named_params_and_buffers(dst_module):
            assert (name in src_tensors) or (not require_all)
            if name in src_tensors:
                tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
    def named_params_and_buffers(self,module):
        assert isinstance(module, torch.nn.Module)
        return list(module.named_parameters()) + list(module.named_buffers())


    
    
def generate_rect_mask( im_size, mask_size, margin=8, rand_mask=True):
    if mask_size==None:
        mask_size=(np.random.randint(int(0.2*im_size[0]),int(0.5*im_size[0])),int(0.2*im_size[1]),int(0.5*im_size[1]))
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)
    if rand_mask:
        sz0, sz1 = mask_size[0], mask_size[1]
        # print(margin, im_size[0] - sz0 - margin)
        # exit()
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
    else:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = (im_size[0] - sz0) // 2
        of1 = (im_size[1] - sz1) // 2
    mask[of0:of0+sz0, of1:of1+sz1] = 1
    mask = np.expand_dims(mask, axis=0)
    
    rect = torch.from_numpy(np.array([[of0, sz0, of1, sz1]], dtype=int))
    return mask, rect
