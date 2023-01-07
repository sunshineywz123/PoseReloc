import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.cuda import amp
import pytorch_lightning as pl
from loguru import logger
from itertools import chain
from src.models.OnePosePlus import OnePosePlus_model
from src.models.OnePosePlus.utils.fine_supervision import fine_supervision
from src.models.OnePosePlus.optimizers.optimizers import (
    build_optimizer,
    build_scheduler,
)
from .losses import Loss
from src.utils.metric_utils import aggregate_metrics, compute_query_pose_errors
from src.utils.comm import gather
from src.utils.plot_utils import draw_reprojection_pair


class PL_OnePosePlus(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.matcher = OnePosePlus_model(self.hparams["OnePosePlus"])
        self.loss = Loss(self.hparams["loss"])
        self.n_vals_plot = max(
            self.hparams["trainer"]["n_val_pairs_to_plot"]
            // self.hparams["trainer"]["world_size"],
            1,
        )

        if self.hparams["pretrained_ckpt"]:
            try:
                self.load_state_dict(
                    torch.load(self.hparams["pretrained_ckpt"], map_location="cpu")[
                        "state_dict"
                    ]
                )
            except RuntimeError as err:
                logger.error(
                    f"Error met while loading pretrained weights: \n{err}\nTry loading with strict=False..."
                )
                self.load_state_dict(
                    torch.load(self.hparams["pretrained_ckpt"], map_location="cpu")[
                        "state_dict"
                    ],
                    strict=False,
                )
            logger.info(
                f"Load '{self.hparams['pretrained_ckpt']}' as pretrained checkpoint"
            )

    def training_step(self, batch, batch_idx):
        self.matcher(batch)

        fine_supervision(batch, self.hparams)

        with amp.autocast(enabled=False):
            self.loss(batch)

        # Update tensorboard on rank0 every n steps
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment[0].add_scalar(f"train/{k}", v, self.global_step)

            self.logger.experiment[0].add_scalar(
                f"train/max conf_matrix", batch["conf_matrix"].max(), self.global_step
            )

            if self.hparams["trainer"]["enable_plotting"]:
                figures = draw_reprojection_pair(batch, visual_color_type="conf")
                for k, v in figures.items():
                    self.logger.experiment[0].add_figure(
                        f"train_match/{k}", v, self.global_step
                    )

        return {"loss": batch["loss"]}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment[0].add_scalar(
                "train/avg_loss_on_epoch", avg_loss, global_step=self.current_epoch
            )

    def validation_step(self, batch, batch_idx):
        self.matcher(batch)

        # Compute metrics
        compute_query_pose_errors(batch, configs=self.hparams["eval_metrics"], training=self.training)
        metrics = {
            "R_errs": batch["R_errs"],
            "t_errs": batch["t_errs"],
            "inliers": batch["inliers"],
        }

        # Visualize match
        val_plot_invervel = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {"evaluation": []}
        if batch_idx % val_plot_invervel == 0:
            figures = draw_reprojection_pair(batch, visual_color_type="conf")

        return {
            "figures": figures,
            "metrics": metrics,
        }

    def validation_epoch_end(self, outputs):
        multi_outputs = (
            [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        )
        multi_val_metrics = []
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.sanity_checking:
                cur_epoch = -1

            def flattenList(x):
                return list(chain(*x))

            # val metrics: dict of list, numpy:
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }

            # figures:
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # tensorboard records only on rank 0:
            if self.trainer.global_rank == 0:
                val_metrics_4tb = aggregate_metrics(
                    metrics, self.hparams["eval_metrics"]["pose_thresholds"]
                )
                for k, v in val_metrics_4tb.items():
                    self.logger.experiment[0].add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                for k, v in figures.items():
                    for plot_idx, fig in enumerate(v):
                        self.logger.experiment[0].add_figure(
                            f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                            fig,
                            cur_epoch,
                            close=True,
                        )

                multi_val_metrics.append(val_metrics_4tb["3cm@3degree"])
            plt.close("all")

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.hparams)
        scheduler = build_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]

