import cv2
import json
import torch
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from src.utils import data_utils


class SPGDataset(Dataset):
    
    def __init__(self, anno_file, pad=True, shape2d=1000, shape3d=2000):
        super(Dataset, self).__init__()

        self.coco = COCO(anno_file)
        self.anns = np.array(self.coco.getImgIds())

        self.pad = pad
        self.shape2d = shape2d
        self.shape3d = shape3d
    
    def read_anno2d(self, anno2d_file, height, width, pad=True):
        """read(and pad) 2d info"""
        with open(anno2d_file, 'r') as f:
            data = json.load(f)
            
        keypoints2d = torch.Tensor(data['keypoints2d']) # [n, 2]
        descriptors2d = torch.Tensor(data['descriptors2d']) # [dim, n]
        scores2d = torch.Tensor(data['scores2d']) # [n, 1]
        assign_matrix = torch.Tensor(data['assign_matrix']) # [2, k]
        
        num_2d_orig = keypoints2d.shape[0]
        
        if pad:
            keypoints2d, descriptors2d, scores2d = data_utils.pad_keypoints2d_random(keypoints2d, descriptors2d, scores2d,
                                                                                     height, width, self.shape2d)
        return keypoints2d, descriptors2d, scores2d, assign_matrix, num_2d_orig
    
    def read_anno3d(self, anno3d_file, pad=True):
        """read(and pad) 3d info"""
        with open(anno3d_file, 'r') as f:
            data = json.load(f)
        
        keypoints3d = torch.Tensor(data['keypoints3d']) # [m, 3]
        descriptors3d = torch.Tensor(data['descriptors3d']) # [dim, m]
        scores3d = torch.Tensor(data['scores3d']) # [m, 1]

        num_3d_orig = keypoints3d.shape[0]

        if pad:
            keypoints3d = data_utils.pad_keypoints3d_random(keypoints3d, self.shape3d)
            descriptors3d, scores3d = data_utils.pad_features3d_random(descriptors3d, scores3d, self.shape3d)

        return keypoints3d, descriptors3d, scores3d, num_3d_orig
         
    def read_anno(self, img_id):
        """
        read image, 2d info and 3d info.
        pad 2d info and 3d info to a constant size.
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]
        
        color_path = self.coco.loadImgs(int(img_id))[0]['img_file']
        image = cv2.imread(color_path)
        height, width, _ = image.shape

        anno3d_file = anno['avg_anno3d_file']
        keypoints3d, descriptors3d, scores3d, num_3d_orig = self.read_anno3d(anno3d_file, pad=self.pad)

        anno2d_file = anno['anno2d_file']
        keypoints2d, descriptors2d, scores2d, assign_matrix, num_2d_orig = self.read_anno2d(anno2d_file, height, width, pad=self.pad) 

        # shape of assign_matrix: 2xk ==> nxm
        conf_matrix = data_utils.reshape_assign_matrix(assign_matrix, num_2d_orig, num_3d_orig,
                                                       self.shape2d, self.shape3d, pad=self.pad)
        
        anno = {
            "keypoints2d": keypoints2d,
            "keypoints3d": keypoints3d,
            "descriptors2d": descriptors2d,
            "descriptors3d": descriptors3d,
            "scores2d": scores2d,
            "scores3d": scores3d,
            "image_size": torch.Tensor([height, width]),
            'image_path': color_path
        }
        return anno, conf_matrix 
    
    def __getitem__(self, index):
        img_id = self.anns[index]
        
        anno, conf_matrix = self.read_anno(img_id)
        return anno, conf_matrix
    
    def __len__(self):
        return len(self.anns)
        