from loguru import logger
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
import os.path as osp
from src.datasets.GATs_loftr_dataset import GATsLoFTRDataset


class GATsLoFTRDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.train_anno_file = kwargs["train_anno_file"]
        self.val_anno_file = kwargs["val_anno_file"]
        assert osp.exists(self.train_anno_file)
        if not osp.exists(self.val_anno_file):
            logger.warning(
                f"Val anno path: {self.val_anno_file} not exists! use train anno instead"
            )
            self.val_anno_file = self.train_anno_file

        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]

        # Data related
        self.train_percent = kwargs["train_percent"]
        self.val_percent = kwargs["val_percent"]
        # 3D part
        self.num_leaf = kwargs["num_leaf"]
        self.shape2d = kwargs["shape2d"]
        self.shape3d = kwargs["shape3d"]
        # 2D part
        self.img_pad = kwargs["img_pad"]
        self.img_resize = kwargs["img_resize"]
        self.df = kwargs["df"]
        self.coarse_scale = kwargs["coarse_scale"]

        # Loader parameters:
        self.train_loader_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.val_loader_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.test_loader_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """ Load data. Set variable: self.data_train, self.data_val, self.data_test"""
        train_set = GATsLoFTRDataset(
            anno_file=self.train_anno_file,
            num_leaf=self.num_leaf,
            img_pad=self.img_pad,
            img_resize=self.img_resize,
            coarse_scale=self.coarse_scale,
            df=self.df,
            shape2d=self.shape2d,
            shape3d=self.shape3d,
            percent=self.train_percent,
            load_pose_gt=True
        )
        print("=> Read train anno file: ", self.train_anno_file)

        val_set = GATsLoFTRDataset(
            anno_file=self.val_anno_file,
            # anno_file=self.train_anno_file,
            num_leaf=self.num_leaf,
            img_pad=self.img_pad,
            img_resize=self.img_resize,
            coarse_scale=self.coarse_scale,
            df=self.df,
            shape2d=self.shape2d,
            shape3d=self.shape3d,
            percent=self.val_percent,
            load_pose_gt=True,
        )

        self.data_train = train_set
        self.data_val = val_set
        self.data_test = val_set

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.train_loader_params)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, **self.val_loader_params)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.test_loader_params)
