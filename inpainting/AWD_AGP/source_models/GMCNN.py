from inpainting.inpainting_gmcnn.pytorch.model.net import InpaintingModel_GMCNN
from inpainting.inpainting_gmcnn.pytorch.options.read_config import GMCNNConfig
from inpainting.inpainting_gmcnn.pytorch.util.utils import generate_rect_mask, generate_stroke_mask, getLatest
import os
import tempfile
from inpainting.AWD_AGP.weight_utils import resolve_weight_path
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import yaml
class GMCNNAPI(torch.nn.Module):
    def __init__(self,dataset='celeba',opt=None):
        super(GMCNNAPI,self).__init__()
        self.dataset=dataset
        self.opt=opt
        config=self.load_Config()
        if dataset!='places2':
            self.module= InpaintingModel_GMCNN(in_channels=4, opt=config,dataset=dataset)
        else:
            self.module= InpaintingModel_GMCNN(in_channels=5, opt=config,dataset=dataset)

        # print('inference',self.module.single_inference)
        # for att in vars(self.module):
        #     print(att)
        # exit()
        self.config=config
        if config.load_model_dir != '' :
            if dataset=='celeba':
                print('Loading pretrained model from {}'.format(config.load_model_dir))
                self.module.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
                print('Loading done.')
            if dataset=='places2':
                checkpoint_dir = resolve_weight_path('weights/inpainting_gmcnn/pytorch/chkpts/places2_rect', self.opt, 'gmcnn_checkpoint_dir', 'AWD_AGP_GMCNN_CKPT_DIR', ['inpainting/inpainting_gmcnn/pytorch/chkpts/places2_rect'], 'GMCNN places2 checkpoint directory')
                print('Loading pretrained model from {}'.format(checkpoint_dir))
                self.module.load_networks(getLatest(os.path.join(checkpoint_dir, '*.pth')))
                print('Loading done.')
            # exit()
    #     print('inference',self.module.single_inference)
    #     exit()
    def forward(self,x,mask,keepFeat=False):
        # print('api',keepFeat)
        # exit()
        original_input=x
        x=x[:,[2,1,0],...]
        # print(x)
        # exit()
        if self.dataset=='celeba':
            result=self.module.single_inference(x, mask)
        elif self.dataset=='places2':
            if keepFeat==False:
                result=self.module.single_inference_fromTF(x, mask,keepFeat=keepFeat)
            else:
                
                result, FeatList=self.module.single_inference_fromTF(x, mask,keepFeat=keepFeat)
        
        # print(len(FeatList))
        # exit()
        result=result[:,[2,1,0],...]
        if self.opt.wo_mask:
            result=result*mask+original_input*(1-mask)
        
        if keepFeat==False:
            return result
        else:
            
            return result, FeatList
    def evaluate(self,*args,**kwargs):
        return self.module.evaluate(*args,**kwargs)

    def load_Config(self):
        config_path = resolve_weight_path('weights/inpainting_gmcnn/pytorch/options/config.yaml', self.opt, 'gmcnn_config', 'AWD_AGP_GMCNN_CONFIG', ['inpainting/inpainting_gmcnn/pytorch/options/config.yaml'], 'GMCNN config')
        config_path_to_load = self._runtime_safe_config(config_path)
        try:
            config=GMCNNConfig(config_path_to_load)
        finally:
            if config_path_to_load != config_path and os.path.exists(config_path_to_load):
                os.unlink(config_path_to_load)
        config.data_file='./imgs/celebahq_256x256/'
        return config

    def _runtime_safe_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        test_dir = config.get('test_dir')
        if not test_dir or os.path.isabs(test_dir):
            if test_dir:
                os.makedirs(test_dir, exist_ok=True)
            return config_path

        runtime_test_dir = os.environ.get(
            'AWD_AGP_GMCNN_TEST_DIR',
            os.path.join(tempfile.gettempdir(), 'awd_agp_gmcnn_test_results'),
        )
        config['test_dir'] = os.path.abspath(runtime_test_dir)
        os.makedirs(config['test_dir'], exist_ok=True)

        handle = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            prefix='awd_agp_gmcnn_',
            delete=False,
        )
        with handle:
            yaml.safe_dump(config, handle, default_flow_style=False)
        return handle.name

def generate_mask_rect(im_shapes, mask_shapes, rand=True):
    mask = np.zeros((im_shapes[0], im_shapes[1])).astype(np.float32)
    if rand:
        of0 = np.random.randint(0, im_shapes[0]-mask_shapes[0])
        of1 = np.random.randint(0, im_shapes[1]-mask_shapes[1])
    else:
        of0 = (im_shapes[0]-mask_shapes[0])//2
        of1 = (im_shapes[1]-mask_shapes[1])//2
    mask[of0:of0+mask_shapes[0], of1:of1+mask_shapes[1]] = 1
    mask = np.expand_dims(mask, axis=2)
    return mask
