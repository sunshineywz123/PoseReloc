import torch
import pytorch_lightning as pl

from src.architectures.GATs_SuperGlue import GATsSuperGlue
from src.losses.focal_loss import FocalLoss


class LitModelGATsSPG(pl.LightningModule):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.architecture = GATsSuperGlue(hparams=self.hparams)
        self.crit = FocalLoss(alpha=1, gamma=2, neg_weights=self.hparams.neg_weights, pos_weights=self.hparams.pos_weights)

        self.train_loss_hist = []
        self.val_loss_hist = []
        self.save_flag = True
    
    def forward(self, x):
        return self.architecture(x)

    def training_step(self, batch, batch_idx):
        self.save_flag = False

        data, conf_matrix_gt = batch
        preds, conf_matrix_pred = self.architecture(data)

        loss_mean = self.crit(conf_matrix_pred, conf_matrix_gt)
        self.log('train/loss', loss_mean, on_step=False, on_epoch=True, prog_bar=False)
        return {'loss': loss_mean, 'preds': preds}
    
    def validation_step(self, batch, batch_idx):
        loss_mean = 0
        preds = None
        self.log('val/loss', loss_mean, on_step=False, on_epoch=True, prog_bar=False)

        return {'loss': loss_mean, 'preds': preds}
    
    def test_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        self.save_flag = True
        self.train_loss_hist.append(self.trainer.callback_metrics['train/loss'])
        self.log('train/loss_best', min(self.train_loss_hist), prog_bar=False)
    
    def validation_epoch_end(self, outputs):
        self.val_loss_hist.append(self.trainer.callback_metrics['val/loss'])
        self.log('val/loss_best', min(self.val_loss_hist), prog_bar=False)
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=self.hparams.milestones,
                                                                gamma=self.hparams.gamma)
            return [optimizer], [lr_scheduler]
        else:
            raise Exception("Invalid optimizer name.")