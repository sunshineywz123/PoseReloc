import torch
import pytorch_lightning as pl
from loguru import logger
from src.architectures.GATs_LoFTR import GATs_LoFTR
from src.architectures.GATs_LoFTR.optimizers.optimizers import build_optimizer, build_scheduler
from src.losses.loftr_loss import LoFTRLoss

class PL_GATsLoFTR(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.matcher = GATs_LoFTR(self.hparams['loftr'])

        self.loss = LoFTRLoss(self.hparams['loss'])

        if self.hparams['pretrained_ckpt']:
            try:
                self.load_state_dict(torch.load(self.hparams['pretrained_ckpt'], map_location='cpu')['state_dict'])
            except RuntimeError as err:
                logger.error(f'Error met while loading pretrained weights: \n{err}\nTry loading with strict=False...')
                self.load_state_dict(torch.load(self.hparams['pretrained_ckpt'], map_location='cpu')['state_dict'], strict=False)
            logger.info(f"Load \'{self.hparams['pretrained_ckpt']}\' as pretrained checkpoint")

        self.train_loss_hist = []
        self.val_loss_hist = []
        self.save_flag = True

    def training_step(self, batch, batch_idx):
        self.matcher(batch)
        
        self.loss(batch)
        pass

    def validation_step(self, batch, batch_idx):
        self.val_loss_hist.append(self.trainer.callback_metrics['val/loss'])
        self.log('val/loss_best', min(self.val_loss_hist), prog_bar=False)

    def training_epoch_end(self, outputs):
        self.save_flag = True
        self.train_loss_hist.append(self.trainer.callback_metrics['train/loss'])
        self.log('train/loss_best', min(self.train_loss_hist), prog_bar=False)

    def validation_epoch_end(self, outputs):
        self.val_loss_hist.append(self.trainer.callback_metrics['val/loss'])
        self.log('val/loss_best', min(self.val_loss_hist), prog_bar=False)
    
    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.hparams)
        scheduler = build_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]

