
def get_superpixel_model(pretrained=None):
    from pathlib import Path
    import torch
    from inpainting.AWD_AGP.superpixel_fcn import models as superpixel_fcn_models

    if pretrained is None:
        repo_root = Path(__file__).resolve().parents[3]
        pretrained = repo_root / "superpixel_fcn" / "pretrain_ckpt" / "SpixelNet_bsd_ckpt.tar"
    pretrained = Path(pretrained)
    if not pretrained.exists():
        raise FileNotFoundError(
            f"Superpixel checkpoint not found: {pretrained}. "
            "Download SpixelNet_bsd_ckpt.tar and pass its path to get_superpixel_model()."
        )
    network_data = torch.load(str(pretrained))
    model = superpixel_fcn_models.__dict__[network_data["arch"]](data=network_data).cuda()
    model.eval()
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    return model

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import time


def get_superpixel(image, model):
    from inpainting.AWD_AGP.superpixel_fcn.train_util import update_spixl_map,shift9pos,get_spixel_image
    downsize=16
      # Data loading code
    input_transform = transforms.Compose([
        # flow_transforms.ArrayToTensor(),
        # transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])
    
    
    # # to get superpixels of the images, we use a pretrained model
    # pretrained='./inpainting/superpixel_fcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar'
    # network_data = torch.load(pretrained)
    # # print("=> using pre-trained model '{}'".format(network_data['arch']))
    # model = models.__dict__[network_data['arch']]( data = network_data).cuda()
    # model.eval()

    # cudnn.benchmark = True


    img_=image
    

    H, W= img_.size()[-2:]
    H_, W_  = int(np.ceil(H/16.)*16), int(np.ceil(W/16.)*16)


    # get spixel id. Resolution of the superpixel map is downscaled by downsize
    n_spixl_h = int(np.floor(H_ / downsize))
    n_spixl_w = int(np.floor(W_ / downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values) # get the shifted coordinators of pixels on 9 directions
    
    # resize the superpixel id map back to the original size of images
    spix_idx_tensor = np.repeat(
      np.repeat(spix_idx_tensor_, downsize, axis=1), downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

    n_spixel =  int(n_spixl_h * n_spixl_w)


    # img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img = F.interpolate(img_, size=( H_,W_), mode='bicubic')[0]
    img_=img_[0]
    
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # compute superpixel output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))
    toc = time.time() - tic
    
    # output=merge_close_superpixel(output,None)

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)

    # resize it to the size of images
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(image.size()[-2:]), mode='nearest').type(torch.int) + 1

    # get the superpixel map visuable
    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1).cuda()
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= n_spixel,  b_enforce_connect=True)

    # merge the closest superpixel together according to the distribution of the pixels inside the superpixel
    old_ori_sz_pixel_map=ori_sz_spixel_map
    edge=None
    while True:
        old_ori_sz_pixel_map=ori_sz_spixel_map
        
        # get the adjecant relationship between each pair of superpixel
        superpixel_graph=get_superpixel_graph(ori_sz_spixel_map)
        # get the feature for each superpixel in the current superpixel map
        superpixel_nodes=get_superpixel_node(ori_sz_spixel_map,image,superpixel_graph)
        # merge the superpixels that are adjcent and have close feature
        ori_sz_spixel_map, edge=merge_close_superpixel(ori_sz_spixel_map,superpixel_graph,superpixel_nodes)
        if torch.sum(torch.abs(old_ori_sz_pixel_map - ori_sz_spixel_map))==0:
            break

    spixel_viz=torch.clamp(image+edge,min=0,max=1)[0].detach().cpu().numpy()
    return ori_sz_spixel_map, spixel_viz


def get_semantic_loss(superpixel_map,mask,w1=100,w2=80,w3=60):

    # only care about superpixel map inside the mask
    masked_superpixel_map=mask*superpixel_map
    diversity=torch.unique(masked_superpixel_map)
    diversity_score=diversity.size()[0]-1

    ######################## get the number of different superpixels covered by the input mask #####################
    h_shift_unit=1
    w_shift_unit=1
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = F.pad(mask, (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit), mode='replicate')
    # input_pd = torch.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:,:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:,:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:,:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:,:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:,:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:,:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:,:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:,:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = torch.cat([     top_left,    top,      top_right,
                                        left,              right,
                                        bottom_left, bottom,    bottom_right], dim=1)


    shift=torch.sum(torch.abs(shift_tensor-center.repeat(1,8,1,1)),dim=1)

    # get the index of superpixels that are on the edges
    edge_diversity=torch.unique((shift>0)*masked_superpixel_map)
    edge_diversity_score=edge_diversity.size()[0]-1
    internal_diversity_score=diversity_score-edge_diversity_score

    # get the isolated area totally inside the mask
    internal_iso_area=masked_superpixel_map.repeat(1,edge_diversity.size()[0],1,1)-edge_diversity.view(1,-1,1,1).repeat(1,1,*(masked_superpixel_map.size()[-2:]))
    internal_iso_area=torch.prod((internal_iso_area!=0),dim=1,keepdim=True)
    internal_iso_area=torch.ones(internal_iso_area.size()).cuda()*(internal_iso_area!=0)

    
    masked_superpixel_map+=0.3*(masked_superpixel_map>0)

    # get the ratio of isolation area over mask area
    internal_iso_area_score=torch.sum(internal_iso_area)/torch.sum(mask)

    if torch.isnan(internal_iso_area_score).any():
        print('in get_semantic_loss', torch.sum(internal_iso_area), torch.sum(mask))
        exit()


    # return the final score and use negative sign since we want to maximize this score to let masks cover more isolated superpixels as possible
    final_score=internal_iso_area_score
    return -final_score

# get the graph (relationship between among superpixel)
def get_superpixel_graph(superpixel_map):
    h_shift_unit=1
    w_shift_unit=1
    # input should be padding as (c, 1+ height+1, 1+width+1)
    
    input_pd = F.pad(superpixel_map.float(), (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit), mode='replicate').int()
    # input_pd = torch.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:,:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:,:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:,:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:,:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:,:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:,:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:,:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:,:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = torch.cat([     top_left,    top,      top_right,
                                        left,              right,
                                        bottom_left, bottom,    bottom_right], dim=1)
    
    # shift=torch.sum(torch.abs(shift_tensor-center.repeat(1,8,1,1)),dim=1)
    
    # get the relationship between each superpixels
    superpixel_graph={}
    for superpixel_id in torch.unique(superpixel_map).tolist():
        
        superpixel_graph[superpixel_id]=torch.unique(torch.masked_select(shift_tensor,(superpixel_map==superpixel_id).repeat(1,8,1,1))).tolist()

        # remove the itself, and collect the neighbor superpixel
        if superpixel_id in superpixel_graph[superpixel_id]:
            superpixel_graph[superpixel_id].remove(superpixel_id)
    # print(superpixel_graph)
    return superpixel_graph


# get the attribute values for each superpixel
def get_superpixel_node(superpixel_map,image,superpixel_graph):

    assert superpixel_map.dim()==4 and image.dim()==4
    superpixel_nodes={}
    for superpixel_id in superpixel_graph:
        # print(image.size(),superpixel_map.size())
        pixels_of_superpixel_id=image*(superpixel_map==superpixel_id)
        variance,mean=torch.var_mean(torch.masked_select(image,(superpixel_map==superpixel_id)))
        size=torch.sum((superpixel_map==superpixel_id)).item() 
        # superpixel_nodes[superpixel_id]=torch.Tensor([mean,variance,size]).cuda()
        superpixel_nodes[superpixel_id]=[torch.Tensor([mean,variance]).cuda(),size]
    
    
    all_feature_vector=torch.stack([feature_vector[0] for i,feature_vector in superpixel_nodes.items()]) # get all the features/attribute values of superpixels


    inter_superpixel_var, inter_superpixel_mean = torch.var_mean(all_feature_vector, dim=0) # cal the distribution of features of all superpixels

    normalized_all_feature_vector=(all_feature_vector-inter_superpixel_mean)/torch.sqrt(inter_superpixel_var) # nromalize the feature for each superpixel

    
    # save it back to the superpxiel node
    for i, superpixel_id in enumerate(superpixel_nodes):
        superpixel_nodes[superpixel_id]=[normalized_all_feature_vector[i],superpixel_nodes[superpixel_id][1]]
        # [normalized feature, size]
    

    return superpixel_nodes




def merge_close_superpixel(superpixel_map,superpixel_graph,superpixel_nodes,thres=0.1):
    new_superpixel_graph={}
    for superpixel_id, superpixel_neighbors in superpixel_graph.items():
        new_superpixel_graph[superpixel_id]=[superpixel_id]
    inv_new_superpixel_graph={}
    for superpixel_id, superpixel_neighbors in superpixel_graph.items():
        inv_new_superpixel_graph[superpixel_id]=superpixel_id

    for superpixel_id in superpixel_graph:
        for neighb_superpixel_id in superpixel_graph[superpixel_id]:

            # if the neighbor superpixel is not in the list of child of the parent of the current superpixel, or the current superpixel is not in the list of child of the parent of the neighbor superpixel 
            if neighb_superpixel_id not in new_superpixel_graph[inv_new_superpixel_graph[superpixel_id]] and superpixel_id not in new_superpixel_graph[inv_new_superpixel_graph[neighb_superpixel_id]]:
                
                # if the superpixel and its neighbor superpixel is close enough
                if compare_superpixel_distance(superpixel_id, neighb_superpixel_id, superpixel_graph, new_superpixel_graph, inv_new_superpixel_graph, superpixel_nodes) <thres: #use the original superpxiel to compare instead of the merged one.

                    # add the neighbor superpixel to its parent superpixel
                    new_superpixel_graph[inv_new_superpixel_graph[superpixel_id]].extend(new_superpixel_graph[inv_new_superpixel_graph[neighb_superpixel_id]])
                    
                    # remove the neigbhor superpixel from its parent
                    superpixel_id_to_remove=inv_new_superpixel_graph[neighb_superpixel_id]

                    # before removing the neighbor superpixel, set parent superpixel of all the child superpixel of the neighbor superpixel as the parent of current pixel. (merging)
                    for child_superpixel_id in new_superpixel_graph[superpixel_id_to_remove]:
                        inv_new_superpixel_graph[child_superpixel_id]=inv_new_superpixel_graph[superpixel_id]
                    
                    # remove the neighbor superpixel, since it has been merged
                    new_superpixel_graph[superpixel_id_to_remove]=[]
                    
    # get the new merged superpixel and get the number of pixels in each merged superpixel in the superpixel map
    merged_superpixel_map=torch.zeros(superpixel_map.size()).cuda()
    num_pixel=0
    for i, superpixel_id in enumerate(new_superpixel_graph):
        for child_superpixel_id in new_superpixel_graph[superpixel_id]:
            merged_superpixel_map+=torch.ones(merged_superpixel_map.size()).cuda()*i*(superpixel_map==child_superpixel_id)
            num_pixel+=torch.sum(superpixel_map==child_superpixel_id)

    # get the edge of the superpixels
    h_shift_unit=1
    w_shift_unit=1
    input_pd = F.pad(merged_superpixel_map.float(), (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit), mode='replicate').int()
    # input_pd = torch.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:,:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:,:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:,:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:,:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:,:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:,:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:,:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:,:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = torch.cat([     top_left,    top,      top_right,
                                        left,              right,
                                        bottom_left, bottom,    bottom_right], dim=1)
    
    shift=torch.sum(torch.abs(shift_tensor-center.repeat(1,8,1,1)),dim=1)
    edge_image=shift>0

    return merged_superpixel_map, edge_image
    
def compare_superpixel_distance(superpixel_id1, superpixel_id2, superpixel_graph, new_superpixel_graph, inv_superpixel_graph, superpixel_nodes):
    features1=[superpixel_nodes[superpixel_id] for superpixel_id in new_superpixel_graph[inv_superpixel_graph[superpixel_id1]]]
    features2=[superpixel_nodes[superpixel_id] for superpixel_id in new_superpixel_graph[inv_superpixel_graph[superpixel_id2]]]
    # print(new_superpixel_graph[inv_superpixel_graph[superpixel_id1]])
    # print(torch.stack([features[0] for features in features1]))
    combined_feature_vector1=torch.matmul(torch.Tensor([features[1] for features in features1]).cuda()/sum([features[1] for features in features1]),torch.stack([features[0] for features in features1]))

    combined_feature_vector2=torch.matmul(torch.Tensor([features[1] for features in features2]).cuda()/sum([features[1] for features in features2]),torch.stack([features[0] for features in features2]))

    # intercluster_var1, intercluster_mean1 = torch.var_mean(torch.stack([features[0] for features in features1]), dim=0)
    # intercluster_var2, intercluster_mean2 = torch.var_mean(torch.stack([features[0] for features in features2]), dim=0)


    # print('mse loss btw feature', F.mse_loss(combined_feature_vector1, combined_feature_vector2))
    return F.mse_loss(combined_feature_vector1, combined_feature_vector2)
