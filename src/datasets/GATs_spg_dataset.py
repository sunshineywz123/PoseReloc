import cv2
try:
    import ujson as json
except ImportError:
    import json
import torch
import numpy as np
import time 

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from src.utils import data_utils


class GATsSPGDataset(Dataset):
    def __init__(self, anno_file, num_leaf, pad=True, shape2d=1000, shape3d=2000):
        super(Dataset, self).__init__()
        
        self.coco = COCO(anno_file)
        self.anns = np.array(self.coco.getImgIds())
        self.num_leaf = num_leaf

        self.pad = pad
        self.shape2d = shape2d
        self.shape3d = shape3d
    
    def read_anno2d(self, anno2d_file, height, width, pad=True):
        """ Read (and pad) 2d info"""
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
    
    def read_anno3d(self, avg_anno3d_file, collect_anno3d_file, idxs_file, pad=True):
        """ Read(and pad) 3d info"""
        with open(avg_anno3d_file, 'r') as f:
            avg_data = json.load(f)
        
        with open(collect_anno3d_file, 'r') as f:
            collect_data = json.load(f)

        idxs = np.load(idxs_file)

        keypoints3d = torch.Tensor(collect_data['keypoints3d']) # [m, 3]
        avg_descriptors3d = torch.Tensor(avg_data['descriptors3d']) # [dim, m]
        collect_descriptors = torch.Tensor(collect_data['descriptors3d']) # [dim, k]
        avg_scores = torch.Tensor(avg_data['scores3d']) # [m, 1]
        collect_scores = torch.Tensor(collect_data['scores3d'])  # [k, 1]

        num_3d_orig = keypoints3d.shape[0]
        if pad:
            keypoints3d = data_utils.pad_keypoints3d_random(keypoints3d, self.shape3d)
            avg_descriptors3d, avg_scores = data_utils.pad_features3d_random(avg_descriptors3d, avg_scores, self.shape3d)
            collect_descriptors, collect_scores = data_utils.build_features3d_leaves(collect_descriptors, collect_scores, idxs,
                                                                                     self.shape3d, num_leaf=self.num_leaf)
        return keypoints3d, avg_descriptors3d, avg_scores, collect_descriptors, collect_scores, num_3d_orig
    
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

        idxs_file = anno['idxs_file']
        avg_anno3d_file = anno['avg_anno3d_file']
        collect_anno3d_file = anno['collect_anno3d_file']
        keypoints3d, avg_descriptors3d, avg_scores3d, clt_descriptors2d, clt_scores2d, num_3d_orig = self.read_anno3d(avg_anno3d_file, collect_anno3d_file, 
                                                                                                                      idxs_file, pad=self.pad)
        anno2d_file = anno['anno2d_file']
        keypoints2d, descriptors2d, scores2d, assign_matrix, num_2d_orig = self.read_anno2d(anno2d_file, height, width,
                                                                                            pad=self.pad)
        
        conf_matrix = data_utils.reshape_assign_matrix(assign_matrix, num_2d_orig, num_3d_orig,
                                                       self.shape2d, self.shape3d, pad=True)

        anno = {
            'keypoints2d': keypoints2d, # [n1, 2]
            'keypoints3d': keypoints3d, # [n2, 3]
            'descriptors2d_query': descriptors2d, # [dim, n1]
            'descriptors3d_db': avg_descriptors3d, # [dim, n2]
            'descriptors2d_db': clt_descriptors2d, # [dim, n2 * num_leaf]
            'image_size': torch.Tensor([height, width]),
        }
        return anno, conf_matrix

    
    def __getitem__(self, index):
        img_id = self.anns[index]

        anno, conf_matrix = self.read_anno(img_id)
        return anno, conf_matrix
    
    def __len__(self):
        return len(self.anns)