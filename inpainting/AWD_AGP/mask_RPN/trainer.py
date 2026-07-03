from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from inpainting.AWD_AGP.mask_RPN.model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator, SemanticAnchorTargetCreator

from torch import nn
import torch as t
from inpainting.AWD_AGP.mask_RPN.utils import array_tool as at
from inpainting.AWD_AGP.mask_RPN.utils.vis_tool import Visualizer

from inpainting.AWD_AGP.mask_RPN.utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

from inpainting.AWD_AGP.mask_RPN.model.utils.superpixel import get_superpixel

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        # 'roi_loc_loss',
                        # 'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn, superpixel_model):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.semantic_based_anchor_target_creator = SemanticAnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

        self.superpixel_model = superpixel_model

    def forward_old(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0] # gt bbox
        label = labels[0] # gt label
        rpn_score = rpn_scores[0] # the confident score for fore and background for all the proposal
        rpn_loc = rpn_locs[0] # all the region proposal without any fliterring
        roi = rois # flittered region proposal, also called ROI. filterred by removing too small regions, regions with small confident scoresa and highly overlapped regions.

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input

        # get the selected roi, including selected foreground rois and background rois.
        # gt_roi_loc is the transformation to the corresponding box of the sampled rois, including background and foreground
        # gt_roi_label is the gt labels for the sampled roi, foreground roi has class label (>0), background roi has label (=0)

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        # roi_cls_loc is the predicted transformation from the sample_roi to the target object in the images. It gt version is gt_roi_loc
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)
        """ roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            all_roi,
            sample_roi_index)   """   
        
        gt_rpn_semantic_score = self.get_semantic_score(rpn_loc, anchor)

        # ------------------ RPN losses -------------------#
        # get the gt bbox transformation from anchor to gt bbox (how the PR should be proposed), as well as gt rpn label (fore or back). Note that the number of gt rpn is same as that of all the rpns
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            # gt_rpn_semantic_score,
            img_size)
        
        # label = 1 means anchor has highest iou with a gt bbox or its iou is larger than the threshold. it may be subsampled if the number is larger than the predefined sample number.
        # label = 0 means anchor has iou that is smaller than the threshold. it may be subsampled if the number is larger than the predefined sample number.
        # label = -1 means anchor is not selected.
        # fast rcnn loc loss between RPN and the gt rpn only if label > 0. It means that the only selected RPN with high enough iou with the gt bbox. 
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        # rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        rpn_cls_loss = F.mse_loss(gt_rpn_semantic_score, rpn_score)

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)
    
    def forward(self, imgs, bboxes, labels, scale):
        # print(imgs.shape, bboxes.shape, labels.shape, scale)
        # print(bboxes[:, :, 2:] - bboxes[:, :, :2])
        # exit()
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0] # gt bbox
        label = labels[0] # gt label
        rpn_score = rpn_scores[0] # the confident score for fore and background for all the proposal
        rpn_loc = rpn_locs[0] # all the region proposal without any fliterring
        roi = rois # flittered region proposal, also called ROI. filterred by removing too small regions, regions with small confident scoresa and highly overlapped regions.

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input

        # get the selected roi, including selected foreground rois and background rois.
        # gt_roi_loc is the transformation to the corresponding box of the sampled rois, including background and foreground
        # gt_roi_label is the gt labels for the sampled roi, foreground roi has class label (>0), background roi has label (=0)

        """ sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        # roi_cls_loc is the predicted transformation from the sample_roi to the target object in the images. It gt version is gt_roi_loc
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index) """
        
        superpixel_map, superpixel_map_vis = get_superpixel(imgs, self.superpixel_model)
        gt_rpn_semantic_score = get_semantic_score(rpn_loc, anchor, superpixel_map).cuda()
        

        # ------------------ RPN losses -------------------#
        # get the gt bbox transformation from anchor to gt bbox (how the PR should be proposed), as well as gt rpn label (fore or back). Note that the number of gt rpn is same as that of all the rpns
        """ gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            # gt_rpn_semantic_score,
            img_size) """
        gt_rpn_loc, gt_rpn_label = self.semantic_based_anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            gt_rpn_semantic_score,
            label,
            img_size)
        
        # label = 1 means anchor has highest iou with a gt bbox or its iou is larger than the threshold. it may be subsampled if the number is larger than the predefined sample number.
        # label = 0 means anchor has iou that is smaller than the threshold. it may be subsampled if the number is larger than the predefined sample number.
        # label = -1 means anchor is not selected.
        # fast rcnn loc loss between RPN and the gt rpn only if label > 0. It means that the only selected RPN with high enough iou with the gt bbox. 
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        # rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        # print('gt_rpn_semantic_score', gt_rpn_semantic_score, 'rpn_score', rpn_score)
        if t.isnan(gt_rpn_semantic_score).any() or t.isnan(rpn_score[:, 0]).any():
            print(t.stack([gt_rpn_semantic_score, rpn_score[:, 0]], dim=1))
            print('score has nan')
            exit()
        rpn_cls_loss = F.mse_loss(gt_rpn_semantic_score.cuda(), rpn_score[:, 0])

        if t.isnan(rpn_locs).any():
            print(t.stack([gt_rpn_semantic_score, rpn_score[:, 0]], dim=1))
            print('loss has nan')
            exit()
        
            

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        """ n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long()) """

        # losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = [rpn_loc_loss, rpn_cls_loss]
        losses = losses + [sum(losses)]

        print('loss', losses)

        return LossTuple(*losses)


    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()


        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

from inpainting.AWD_AGP.mask_RPN.model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from inpainting.AWD_AGP.mask_RPN.model.utils.superpixel import get_semantic_loss
def get_semantic_score(rpn_loc, anchor, superpixel_map, relative_mask_size=0.33):
    # print((anchor).shape, (rpn_loc).shape)
    rpn_box = loc2bbox(anchor, rpn_loc.cpu().detach().numpy()) 
    # print('anchor', anchor)
    """ Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`. """
    
    all_semantic_loss = []
    for index_rpn_box in range(rpn_box.shape[0]):
        # rpn_box = rpn_box.clone()
        # rpn_box[:, 2] = rpn_box[:, 2] - rpn_box[:, 0]
        # rpn_box[:, 3] = rpn_box[:, 3] - rpn_box[:, 1]
        # assert t.equal(rpn_box[:, 2:] - 84, t.zeros_like(rpn_box[:, 2:]).cuda())
        if np.isnan(rpn_box[index_rpn_box, :2]).any():    
            print('corresponding rpn box', rpn_box[index_rpn_box], rpn_loc[index_rpn_box])
            exit()
        mask, _ = generate_rect_mask(superpixel_map.size()[-2:], mask_size=(int(superpixel_map.shape[-2]*relative_mask_size), int(superpixel_map.shape[-1]*relative_mask_size)), position=rpn_box[index_rpn_box, :2] ,rand_mask=False, batch_size=superpixel_map.size()[0])# it needs to be aligned with mask size of (84, 84)

        if t.sum(mask) == 0:
            print('rpn_box[index_rpn_box, :2]', rpn_box[index_rpn_box, :2])
            all_semantic_loss.append(0)
            continue

        # print('rpn_box', rpn_box[:10])
        # print('rpn_loc', rpn_loc[:10])
        # print('anchor', anchor[:10])
        # print(rpn_box[index_rpn_box, 2].item() - rpn_box[index_rpn_box, 0].item() , rpn_box[index_rpn_box, 3].item() - rpn_box[index_rpn_box, 1].item()  )
        assert round(rpn_box[index_rpn_box, 2].item() - rpn_box[index_rpn_box, 0].item()) == 84 and round(rpn_box[index_rpn_box, 3].item() - rpn_box[index_rpn_box, 1].item()) == 84

        # print(superpixel_map.shape, mask.shape)
        semantic_loss = - get_semantic_loss(superpixel_map, mask)
        
        if t.isnan(semantic_loss).any():
            print('semantic_loss is nan', t.isnan(superpixel_map).any(), t.isnan(mask).any()
                  )
            exit()

        all_semantic_loss.append(semantic_loss)
    return t.tensor(all_semantic_loss)
        

import numpy as np
def generate_rect_mask(im_size, mask_size, margin=8, rand_mask=False, position = 'center', batch_size=None):
    # initialize the mask with a default size if there is not input mask size
    if mask_size==None:
        mask_size=(np.random.randint(int(0.2*im_size[0]),int(0.5*im_size[0])),int(0.2*im_size[1]),int(0.5*im_size[1]))
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)


    if rand_mask:
        # randomly initialize the mask
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
    else:
        """ if position == 'center':
            # initialize the mask at the center of the images
            sz0, sz1 = mask_size[0], mask_size[1]
            of0 = (im_size[0] - sz0) // 2
            of1 = (im_size[1] - sz1) // 2
        elif position == 'left_top':
            # initialize the mask at the left top of the images
            sz0, sz1 = mask_size[0], mask_size[1]
            of0 = 0
            of1 = 0
        else: """
        # initialize the mask at the input position
        of0 = max(int(position[0]),0)
        of1 = max(int(position[1]),0)
        sz0, sz1 = int(min(mask_size[0],im_size[0] - position[0])), int(min(mask_size[1],im_size[1] - position[1]))

    # print('of0', of0, of0+sz0, of1, of1+sz1, mask_size[0], im_size[0] - position[0])
    mask[of0:of0+sz0, of1:of1+sz1] = 1

    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = np.repeat(mask,batch_size,axis=0)
    mask = t.from_numpy(mask).cuda()
    rect = t.from_numpy(np.array([[of0, sz0, of1, sz1]], dtype=int))

    rect_list = []
    return mask, rect