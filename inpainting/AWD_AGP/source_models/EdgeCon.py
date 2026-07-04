import argparse
from inpainting.edge_connect.src.edge_connect import EdgeConnect
import torch
from inpainting.edge_connect.src.config import Config
from shutil import copyfile
import os
from inpainting.AWD_AGP.weight_utils import resolve_weight_path

import kornia as K
import torchvision

class ConnectEdgeAPI(torch.nn.Module):
    def __init__(self,opt):
        super(ConnectEdgeAPI,self).__init__()
        self.opt=opt
        config=self.load_config(2)
        # print(config.PATH)
        # exit()
        self.edgeconnect=EdgeConnect(config)
        self.edgeconnect.load()
    def forward(self,images,masks,keepFeat=False):
        images=images[:,[2,1,0],...]
        images=images*(1-masks)+masks
        images_gray=torchvision.transforms.functional.rgb_to_grayscale(images)
        # print(images_gray)

        # edgeMask=  (1 - masks )
    
        edges=K.filters.canny(images_gray,sigma=(2,2),low_threshold=0.2,high_threshold=0.3,kernel_size=(21,21))[1]
        
        # edges=torchvision.transforms.functional.rgb_to_grayscale(edges)
        edges=edges*(1 - masks )
        
        def binary_erosion(maskedge,kernel_size):
            # maskedge=(1-maskedge)
            kernel=torch.ones((1,1,kernel_size,kernel_size)).cuda()
            result=torch.clamp(torch.nn.functional.conv2d(maskedge,kernel,padding='same'),0,1)
            # result=(1-result)
            return result
        # maskedges=K.filters.canny(masks.unsqueeze(0),sigma=(2,2),kernel_size=(11,11))[1]
        maskedges=K.filters.canny(masks ,sigma=(2,2),kernel_size=(11,11))[1]
        maskedges=binary_erosion(maskedges,3)
        edges=edges*(1-maskedges)


        if self.edgeconnect.config.MODEL==3:
            inputs = (images * (1 - masks)) + masks
            
            if keepFeat==False:
                # outputs = self.edgeconnect.edge_model(images_gray, edges, masks,keepFeat=keepFeat).detach()
                outputs = self.edgeconnect.edge_model(images_gray, edges, masks,keepFeat=keepFeat)
            else:
                outputs, FeatList1 = self.edgeconnect.edge_model(images_gray, edges, masks,keepFeat=keepFeat)
            edges = (outputs * masks + edges * (1 - masks)).detach()
            if keepFeat==False:
                outputs = self.edgeconnect.inpaint_model(inputs, edges, masks,keepFeat=keepFeat)
            else:
                outputs, FeatList  = self.edgeconnect.inpaint_model(inputs, edges, masks,keepFeat=keepFeat)
            
            if self.opt.wo_mask:
                outputs_merged=outputs
            else:
                outputs_merged = (outputs * masks) + (images * (1 - masks))

        outputs_merged=outputs_merged[:,[2,1,0],...]

        if keepFeat==False:
            return outputs_merged
        else:
            return outputs_merged, FeatList
    
    # def forward(self,images,images_gray,edges,masks):
    #     images=images[:,[2,1,0],...]
        
    #     edges=torchvision.transforms.functional.rgb_to_grayscale(images)
    #     # print(images.size(),torch.max(images_gray),torch.max(edges),torch.max(masks))
    #     # exit()
    #     if self.edgeconnect.config.MODEL==3:
    #         inputs = (images * (1 - masks)) + masks
    #         outputs = self.edgeconnect.edge_model(images_gray, edges, masks).detach()
    #         edges = (outputs * masks + edges * (1 - masks)).detach()
    #         outputs = self.edgeconnect.inpaint_model(images, edges, masks)
    #         outputs_merged = (outputs * masks) + (images * (1 - masks))
    #     return outputs_merged
    
    


    def load_config(self,mode=None):
        r"""loads model config

        Args:
            mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
        """

        # parser = argparse.ArgumentParser()
        # parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
        # parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

        # # test mode
        # if mode == 2:
        #     parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        #     parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        #     parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        #     parser.add_argument('--output', type=str, help='path to the output directory')

        # args = parser.parse_args()

        if self.opt.dataset=='celeba':
            config_path = resolve_weight_path('weights/edge_connect/checkpoints/celeba/pipeline_config.yml', self.opt, 'edgeconnect_config', 'AWD_AGP_EDGECONNECT_CONFIG', ['inpainting/edge_connect/checkpoints/celeba/pipeline_config.yml'], 'EdgeConnect celeba config')
        elif self.opt.dataset=='places2':
            config_path = resolve_weight_path('weights/edge_connect/checkpoints/places2/pipeline_config.yml', self.opt, 'edgeconnect_config', 'AWD_AGP_EDGECONNECT_CONFIG', ['inpainting/edge_connect/checkpoints/places2/pipeline_config.yml'], 'EdgeConnect places2 config')
        

        # create checkpoints path if does't exist
        # if not os.path.exists(args.path):
        #     os.makedirs(args.path)

        # copy config template if does't exist
        if not os.path.exists(config_path):
            copyfile('inpainting/edge_connect/config.yml.example', config_path)

        # load config file
        config = Config(config_path)

        config.DEVICE = torch.device("cuda")
        # train mode
        if mode == 1:
            config.MODE = 1
            # if args.model:
            #     config.MODEL = args.model

        # test mode
        elif mode == 2:
            config.MODE = 2
            config.MODEL = 3
            config.INPUT_SIZE = 0

            # if args.input is not None:
            #     config.TEST_FLIST = args.input

            # if args.mask is not None:
            #     config.TEST_MASK_FLIST = args.mask

            # if args.edge is not None:
            #     config.TEST_EDGE_FLIST = args.edge

            # if args.output is not None:
            #     config.RESULTS = args.output

        # # eval mode
        # elif mode == 3:
        #     config.MODE = 3
        #     config.MODEL = args.model if args.model is not None else 3

        return config
