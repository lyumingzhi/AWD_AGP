from inpainting.crfill.models.inpaint_model import InpaintModel
import torchvision.transforms as transforms
import yaml 
import os 
from inpainting.AWD_AGP.weight_utils import resolve_weight_path
import torch

class CrfillAPI(torch.nn.Module):
    def __init__(self,dataset,opt):
        super(CrfillAPI,self).__init__()
        self.opt=opt

        self.dataset=dataset
        opt=self.load_config(opt)
        self.module=InpaintModel(opt)
        transform_list = [
                # transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                ]
        self.image_transform = transforms.Compose(transform_list)
        # self.mask_transform = transforms.Compose([
        #     transforms.ToTensor()
        #     ])
    def forward(self,x,mask,keepFeat=False):
        x=x.clone()[:,[2,1,0],...]
        x=self.image_transform(x)
        data={'image':x,'mask':mask}
        # assert cv2.imwrite('example_gt.jpg',(data['image'])[0].detach().cpu().numpy().transpose(1,2,0)[...,::-1]*255)
        # assert cv2.imwrite('example_input.jpg',(data['image']*(1-mask))[0].detach().cpu().numpy().transpose(1,2,0)[...,::-1]*255)
        if keepFeat==False:
            composed_image, inputs=self.module( data, 'inference',keepFeat=keepFeat,wo_mask=self.opt.wo_mask)
        else:
            composed_image, inputs, FeatList=self.module( data, 'inference',keepFeat=keepFeat,wo_mask=self.opt.wo_mask)

        # print(len(FeatList))
        # exit(0)
        composed_image=torch.clamp(composed_image,-1,1)/2+0.5
        # return composed_image, inputs

        composed_image=composed_image.clone()[:,[2,1,0],...]

        if keepFeat==False:
            return composed_image
        else:
            return composed_image,FeatList
    
    def load_config(self, opt=None):
        if self.dataset=='objrmv':
            config_path = resolve_weight_path(
                'weights/crfill/checkpoints/objrmv/objrmv.yaml',
                opt,
                'crfill_objrmv_config',
                'AWD_AGP_CRFILL_OBJRMV_CONFIG',
                ['inpainting/crfill/checkpoints/objrmv/objrmv.yaml'],
                'CR-Fill objrmv config',
            )
        elif self.dataset=='places2':
            config_path = resolve_weight_path(
                'weights/crfill/checkpoints/places/places.yaml',
                opt,
                'crfill_places_config',
                'AWD_AGP_CRFILL_PLACES_CONFIG',
                ['inpainting/crfill/checkpoints/places/places.yaml'],
                'CR-Fill places config',
            )
        else:
            raise ValueError(f'Unsupported CR-Fill dataset: {self.dataset}')
        config=Crfill_Config(config_path)
        return config



class Crfill_Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['Conifg_PATH'] = os.path.dirname(config_path)

        # self.parse()
        # print(self)
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        # if DEFAULT_CONFIG.get(name) is not None:
        #     return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')

    def gather_options(self):
        # initialize parser with basic options
        

        # get the basic options
        # opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        # model_name = opt.model
        # model_option_setter = models.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options



        # if there is opt_file, load it.
        # The previous default options will be overwritten
        # if opt.load_from_opt_file:
        #     parser = self.update_options_from_file(parser, opt)

        # opt = parser.parse_args()
        # self.parser = parser
        return 
    def parse(self):
        # opt = self.gather_options()
        # opt.isTrain = self.isTrain   # train or test

        self.print()
    

        # set gpu ids
        str_ids = self.gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])

        assert len(self.gpu_ids) == 0 or self.batchSize % len(self.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (self.batchSize, len(self.gpu_ids))

        # self.opt = opt
        # return self.opt
