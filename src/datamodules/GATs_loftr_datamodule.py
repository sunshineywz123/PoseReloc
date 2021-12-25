from re import M
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from src.datasets.GATs_loftr_dataset import GATsLoFTRDataset


class GATsLoFTRDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.anno_file = kwargs["anno_file"]
        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]

        # Data related
        # 3D part
        self.num_leaf = kwargs["num_leaf"]
        self.shape2d = kwargs["shape2d"]
        self.shape3d = kwargs["shape3d"]
        # 2D part
        self.img_pad = kwargs["img_pad"]
        self.img_resize = kwargs["img_resize"]
        self.df = kwargs["df"]
        self.coarse_scale = kwargs["coarse_scale"]

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """ Load data. Set variable: self.data_train, self.data_val, self.data_test"""
        trainset = GATsLoFTRDataset(
            anno_file=self.anno_file,
            num_leaf=self.num_leaf,
            img_pad=self.img_pad,
            img_resize=self.img_resize,
            coarse_scale=self.coarse_scale,
            df=self.df,
            shape2d=self.shape2d,
            shape3d=self.shape3d,
        )
        print("=> Read anno file: ", self.anno_file)

        self.data_train = trainset
        self.data_val = trainset
        self.data_test = trainset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
