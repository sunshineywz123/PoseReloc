from loguru import logger

import torch
import torch.nn as nn


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # coarse
        self.correct_thr = config["fine_correct_thr"]  # 1 represents within window
        self.c_pos_w = config["pos_weight"]
        self.c_neg_w = config["neg_weight"]
        # fine
        self.fine_type = config["fine_type"]

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        TODO: Refactor
        """
        if self.config["coarse_type"] == "cross_entropy":
            # assert not self.config['spg_spvs']
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = -torch.log(conf[conf_gt == 1])
            loss_neg = -torch.log(1 - conf[conf_gt == 0])
            if weight is not None:
                loss_pos = loss_pos * weight[conf_gt == 1]
                loss_neg = loss_neg * weight[conf_gt == 0]
            return self.c_pos_w * loss_pos.mean() + self.c_neg_w * loss_neg.mean()
        elif self.config["coarse_type"] == "focal":
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.config["focal_alpha"]
            gamma = self.config["focal_gamma"]

            if self.config['spg_spvs']:
                pos_conf = conf[:, :-1, :-1][conf_gt == 1] if not self.config['dual_softmax'] \
                            else conf[conf_gt == 1]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                if not self.config['dual_softmax']:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out, but only through setting regions in sim_matrix to -inf.
                    loss_pos = loss_pos * weight[conf_gt == 1]
                    if not self.config['dual_softmax']:
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                return self.c_pos_w * loss_pos.mean() + self.c_neg_w * loss_neg.mean() if not self.config['dual_softmax'] \
                        else self.c_pos_w * loss_pos.mean()
                # positive and negative elements have similar loss weights. => more balanced loss weight
            else:
                loss_pos = (
                    -alpha
                    * torch.pow(1 - conf[conf_gt == 1], gamma)
                    * (conf[conf_gt == 1]).log()
                )
                loss_neg = (
                    -(1 - alpha)
                    * torch.pow(conf[conf_gt == 0], gamma)
                    * (1 - conf[conf_gt == 0]).log()
                )
                if weight is not None:
                    loss_pos = loss_pos * weight[conf_gt == 1]
                    loss_neg = loss_neg * weight[conf_gt == 0]
                
                if loss_pos.shape[0] == 0:
                    logger.warning('len of loss pos is zero!')
                    loss_mean = self.c_neg_w * loss_neg.mean()
                elif loss_neg.shape[0] == 0:
                    logger.warning('len of loss neg is zero!')
                    loss_mean = self.c_pos_w * loss_pos.mean()
                else:
                    loss_pos_mean = loss_pos.mean()
                    loss_neg_mean = loss_neg.mean()
                    loss_mean = self.c_pos_w * loss_pos_mean + self.c_neg_w * loss_neg_mean
                
                return loss_mean
                # each negative element has smaller weight than positive elements. => higher negative loss weight
        else:
            raise KeyError

    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == "l2_with_std":
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == "l2":
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = (
            torch.linalg.norm(expec_f_gt, ord=float("inf"), dim=1) < self.correct_thr
        )
        if correct_mask.sum() == 0:
            if (
                self.training
            ):  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = (
            torch.linalg.norm(expec_f_gt, ord=float("inf"), dim=1) < self.correct_thr
        )

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1.0 / torch.clamp(std, min=1e-10)
        weight = (
            inverse_std / torch.mean(inverse_std)
        ).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if correct_mask.sum() == 0:
            if (
                self.training
            ):  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 1e-6
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(
            -1
        )
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    def compute_coarse_prior_loss(
        self, prior0, prior1, conf_gt, weight0=None, weight1=None
    ):
        """
        Args:
            prior0 (torch.Tensor): (N, HW0, 1)
            prior1 (torch.Tensor): (N, HW1, 1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight{i} (torch.Tensor): (N, H{i}, W{i})
        """
        prior0_gt = conf_gt.sum(-1) != 0
        prior1_gt = conf_gt.sum(1) != 0

        prior0, prior1 = prior0[..., 0], prior1[..., 0]
        prior, prior_gt = map(
            lambda x: torch.cat(x, 0), [[prior0, prior1], [prior0_gt, prior1_gt]]
        )
        alpha, gamma = (
            self.config["coarse_prior"]["focal_alpha"],
            self.config["coarse_prior"]["focal_gamma"],
        )

        loss_pos = (
            -alpha * torch.pow(1 - prior[prior_gt], gamma) * (prior[prior_gt]).log()
        )
        loss_neg = (
            -alpha * torch.pow(prior[~prior_gt], gamma) * (1 - prior[~prior_gt]).log()
        )
        if weight0 is not None:
            weight = torch.cat([weight0.flatten(-2), weight1.flatten(-2)], 0)
            loss_pos = loss_pos * weight[prior_gt]
            loss_neg = loss_neg * weight[~prior_gt]
        loss = loss_pos.mean() + loss_neg.mean()
        return loss

    def compute_fine_rejection_loss(self, conf_f, expec_f_gt):
        """
        Args:
            conf_f (torch.Tensor): [M] sigmoid output
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        alpha, gamma = (
            self.config["fine_rejection"]["focal_alpha"],
            self.config["fine_rejection"]["focal_gamma"],
        )
        # conf_f = torch.clamp(conf_f, 1e-6, 1-1e-6)
        if expec_f_gt.shape[0] == 0:
            logger.warning(
                "No input for fine-level rejection, might cause ddp deadlock."
            )
            return None

        pos_mask = torch.linalg.norm(expec_f_gt, ord=float("inf"), dim=1) < 1
        loss_pos = (
            -alpha * torch.pow(1 - conf_f[pos_mask], gamma) * (conf_f[pos_mask]).log()
        )
        loss_neg = (
            -alpha * torch.pow(conf_f[~pos_mask], gamma) * (1 - conf_f[~pos_mask]).log()
        )
        loss = loss_pos.mean() + loss_neg.mean()

        return loss

    @torch.no_grad()
    def compute_c_weight(self, data):
        if "mask0" in data:
            c_weight = (
                data["mask0"].flatten(-2)[..., None]
                * data["mask1"].flatten(-2)[:, None]
            )
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data["conf_matrix"], data["conf_matrix_gt"], weight=c_weight
        )
        loss = loss_c * self.config["coarse_weight"]
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        if 'expec_f' in data:
            loss_f = self.compute_fine_loss(data["expec_f"], data["expec_f_gt"])
            if loss_f is not None:
                loss += loss_f * self.config["fine_weight"]
                loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})
            else:
                assert self.training is False
                loss_scalars.update({"loss_f": torch.tensor(1.0)})  # 1 is the upper bound

        # 3. fine-level rejection loss
        # if self.config['fine_rejection']['enable']:
        #     loss_f_rej = self.compute_fine_rejection_loss(data['conf_f'], data['expec_f_gt'])
        #     if loss_f_rej is not None:
        #         loss += loss_f_rej  # No additional weight except for focal_alpha
        #         loss_scalars.update({'loss_f_rej': loss_f_rej.clone().detach().cpu()})
        #     else:
        #         loss_scalars.update({'loss_f_rej': torch.tensor(-1.)})

        # 4. coarse-level prior loss
        # if self.config['coarse_prior']['enable']:
        #     weight0 = data['mask0'] if 'mask0' in data else None
        #     weight1 = data['mask1'] if 'mask0' in data else None
        #     loss_c_prior = self.compute_coarse_prior_loss(data['prior0'], data['prior1'], data['conf_matrix_gt'], weight0=weight0, weight1=weight1)
        #     loss += loss_c_prior
        #     loss_scalars.update({'loss_c_prior': loss_c_prior.clone().detach().cpu()})

        loss_scalars.update({"loss": loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
