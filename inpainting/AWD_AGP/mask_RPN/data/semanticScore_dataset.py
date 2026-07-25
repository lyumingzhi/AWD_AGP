import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image

from pathlib import Path
import json

import random
import torchvision.transforms as transforms

class SemanticScore_Dataset:

    # def __init__(self, data_dir, split='trainval',
    #              use_difficult=False, return_difficult=False,
    #              ):
    def __init__(self,
                 image_dir=None, scores_dir=None,
                 use_difficult=False, return_difficult=False,
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        # id_list_file = os.path.join(
        #     data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        # self.ids = [id_.strip() for id_ in open(id_list_file)]
        # self.data_dir = data_dir
        # self.use_difficult = use_difficult
        # self.return_difficult = return_difficult
        # self.label_names = VOC_BBOX_LABEL_NAMES

        image_dir = image_dir or os.environ.get('AWD_AGP_RPN_IMAGE_DIR')
        scores_dir = scores_dir or os.environ.get('AWD_AGP_RPN_SCORES_DIR')
        if not image_dir or not os.path.isdir(image_dir):
            raise FileNotFoundError('RPN semantic image directory not found. Pass image_dir or set AWD_AGP_RPN_IMAGE_DIR.')
        if not scores_dir or not os.path.isdir(scores_dir):
            raise FileNotFoundError('RPN semantic scores directory not found. Pass scores_dir or set AWD_AGP_RPN_SCORES_DIR.')

        image_files = os.listdir(image_dir)
        self.image_files = [os.path.join(image_dir, filename) for filename in image_files]

        self.ids = []
        self.scores_files = []
        for image_filename in image_files:
            id = Path(image_filename).stem
            score_file_path = os.path.join(scores_dir, id+'.json')
            if not os.path.exists(score_file_path):
                raise FileNotFoundError(f'RPN semantic score file not found: {score_file_path}')
            self.scores_files.append(score_file_path)
            self.ids.append(id)

        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        
        self.num_bbox = 10
        
        self.label_names = 'Semantic Score'

        self.transformer =transforms.Compose([
                            transforms.ToTensor(),
                            transforms.CenterCrop((256,256)),
                            transforms.Resize((256,256))
                        ])
        
    def __len__(self):
        # return len(self.ids)
        return len(self.image_files)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """

        # Load a image
        img_file = self.image_files[i]
        img = read_image(img_file, color=True)

        img = self.transformer(img.transpose(1,2,0))

        # print('image', img.shape)
        # exit()


        id_ = self.ids[i]
        # anno = ET.parse(
        #     os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))

        with open(self.scores_files[i]) as f:
            score_records = json.load(f)
        bbox = list()
        label = list()
        difficult = list()

        num_scores = len(score_records['score_list'])
        assert len(score_records['score_list']) == len(score_records['box_list'])

        all_bbox_list = score_records['box_list'][::-1]
        all_score_list = score_records['score_list'][::-1]
        seen_boxes = set()
        index = 0
        while len(bbox) != self.num_bbox and index < len(all_bbox_list):
            raw_box = tuple(all_bbox_list[index])
            if raw_box not in seen_boxes:
                seen_boxes.add(raw_box)
                bbox.append((all_bbox_list[index][0], all_bbox_list[index][1], all_bbox_list[index][0]+int(img.shape[-2] * all_bbox_list[index][2]), all_bbox_list[index][1]+int(img.shape[-1] * all_bbox_list[index][3]))) # 'ymin', 'xmin', 'ymax', 'xmax'

                label.append(all_score_list[index])

                difficult.append(1)
            index += 1

        if len(bbox) < self.num_bbox:
            raise ValueError(f'Not enough unique RPN boxes in {self.scores_files[i]}: expected {self.num_bbox}, got {len(bbox)}')

        for i in range(self.num_bbox):
            index = random.randint(len(all_bbox_list)//2, len(all_bbox_list)-1)
            raw_box = tuple(all_bbox_list[index])
            if raw_box not in seen_boxes:
                seen_boxes.add(raw_box)
                bbox.append((all_bbox_list[index][0], all_bbox_list[index][1], all_bbox_list[index][0]+int(img.shape[-2] * all_bbox_list[index][2]), all_bbox_list[index][1]+int(img.shape[-1] * all_bbox_list[index][3]))) # 'ymin', 'xmin', 'ymax', 'xmax'

                label.append(all_score_list[index])

                difficult.append(1)
        
        """ for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name)) """
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        
        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example
