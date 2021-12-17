import os
import random
import os.path as osp
import torch

from torch.utils.data import Dataset
from src.datasets.utils import (
    read_grayscale,
    load_intrinsics_from_h5,
)


class loftr_coarse_dataset(Dataset):
    """Build image Matching Challenge Dataset image pair (val & test)"""

    def __init__(
        self,
        args,
        image_lists,
        covis_pairs,
    ):
        """
        Parameters:
        ---------------
        """
        super().__init__()
        self.img_dir = image_lists
        self.img_resize = args['img_resize']
        self.df = args['df']
        n_imgs = args['n_imgs']
        shuffle = args['shuffle']

        # Load pairs: 
        with open(covis_pairs, 'r') as f:
            self.pair_list = f.read().rstrip('\n').split('\n')

        all_img_names = sorted(os.listdir(self.img_dir))
        img_names = all_img_names[:n_imgs]

        self.img_names = sorted(img_names, reverse=True)  # sorted
        self.img_paths = [
            osp.join(self.img_dir, img_name) for img_name in self.img_names
        ]

        self.f_names = [osp.splitext(n)[0] for n in self.img_names]

        if shuffle:
            random.shuffle(self.pair_list)

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        return self._get_single_item(idx)

    @classmethod
    def _get_sliced_instance(cls, other, _slice):
        obj = cls.__new__(cls)
        super(loftr_coarse_dataset, obj).__init__()
        # TODO: It's better to update f_names, img_scales and pair_ids
        obj.f_names = other.f_names
        obj.img_scales = other.img_scales
        obj.pair_ids = other.pair_ids[_slice]
        return obj

    def _get_single_item(self, idx):
        img_path0, img_path1 = self.pair_list[idx].split(' ')
        img_scale0 = read_grayscale(
            osp.join(self.img_dir, img_path0),
            (self.img_resize,),
            df=self.df,
            ret_scales=True,
        )
        img0, scale0 = map(lambda x: x[None], img_scale0)  # no dataloader operation
        img_scale1 = read_grayscale(
            osp.join(self.img_dir, img_path1),
            (self.img_resize,),
            df=self.df,
            ret_scales=True,
        )
        img1, scale1 = map(lambda x: x[None], img_scale1)  # no dataloader operation

        data = {
            "image0": img0,  # 1*1*H*W because no dataloader operation, if batch: 1*H*W
            "image1": img1,
            "scale0": scale0,  # 1*2
            "scale1": scale1,
            "f_name0": osp.basename(img_path0).rsplit('.', 1)[0],
            "f_name1": osp.basename(img_path1).rsplit('.', 1)[0],
            "frameID": idx,
            # "img_path": [osp.join(self.img_dir, img_name)]
            "pair_key": (img_path0, img_path1),
        }

        return data

