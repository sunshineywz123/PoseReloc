from itertools import chain

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
from kornia import homography_warp
from kornia.geometry.warp import normalize_homography, normal_transform_pixel
from src.utils.profiler import PassThroughProfiler

from .backbone import (
    build_backbone,
    _extract_backbone_feats,
    _split_backbone_feats,
    _get_feat_dims,
)
from .loftr_module import (
    LocalFeatureTransformer,
    FinePreprocess,
    build_rejector,
    build_coarse_prior,
)
from .utils.position_encoding import PositionEncodingSine
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .utils.selective_kernel import build_ssk_merge
from .utils.guided_matching_fine import build_guided_matching
from .utils.sample_feature_from_featuremap import sample_feature_from_featuremap
from src.utils.misc import upper_config  # TODO: Remove the out of package import
from src.utils.torch_utils import torch_speed_test


class LoFTR_SfM(nn.Module):
    def __init__(self, config={}, profiler=None, debug=False, extract_coarse_feats_mode=False):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.debug = debug
        self.enable_fine_loftr = config["loftr_fine"]["enable"]

        # Modules
        self.backbone = build_backbone(config["loftr_backbone"])
        self.pos_encoding = PositionEncodingSine(
            config["loftr_coarse"]["d_model"],
            max_shape=config["loftr_coarse"]["pos_emb_shape"],
        )
        self.loftr_coarse = LocalFeatureTransformer(config["loftr_coarse"])
        self.coarse_matching = CoarseMatching(
            config["loftr_match_coarse"],
            config["loftr_match_fine"]["detector"],
            profiler=self.profiler,
        )
        # TODO: Multi-Scale DeiT Feature for Fine-Level
        self.fine_preprocess = FinePreprocess(
            config["loftr_fine"],
            config["loftr_coarse"]["d_model"],
            cf_res=config["loftr_backbone"]["resolution"],
            feat_ids=config["loftr_backbone"]["resnetfpn"]["output_layers"],
            feat_dims=_get_feat_dims(config["loftr_backbone"]),
        )

        # # TODO: remove! not useful
        self.loftr_fine = LocalFeatureTransformer(config["loftr_fine"])
        self.fine_matching = FineMatching(
            config["loftr_match_fine"], _full_cfg=upper_config(config)
        )
        # Optional Modules
        self.coarse_prior = build_coarse_prior(config["loftr_coarse"])
        self.fine_rejector = build_rejector(config["loftr_fine"])
        self.coarse_ssk_merge = build_ssk_merge(config["loftr_coarse"])
        self.guided_matching = build_guided_matching(config["loftr_guided_matching"])

        # self.pose_depth_refinement = PoseDepthRefinement(config['loftr_sfm'])

        # TODO: this should be removed in the future @zehong
        # fixed pretrained coarse weights
        self.loftr_coarse_pretrained = config["loftr_coarse"]["pretrained"]
        if self.loftr_coarse_pretrained is not None:
            ckpt = torch.load(self.loftr_coarse_pretrained, "cpu")["state_dict"]
            for k in list(ckpt.keys()):
                if "loftr_coarse" in k:
                    newk = k[k.find("loftr_coarse") + len("loftr_coarse") + 1 :]
                    ckpt[newk] = ckpt[k]
                if "coarse_matching" in k:
                    newk = k[k.find("coarse_matching") + len("coarse_matching") + 1 :]
                    self.coarse_matching.load_state_dict({newk: ckpt[k]})
                    self.coarse_matching.requires_grad_(False)
                ckpt.pop(k)
            self.loftr_coarse.load_state_dict(ckpt)
            for param in self.loftr_coarse.parameters():
                param.requires_grad = False

        # Disable grads when use gt mode (for convenience, inference without backprop, but better to disable)
        if config["loftr_match_coarse"]["_gt"]:
            for param in self.loftr_coarse.parameters():
                param.requires_grad = False
        if config["loftr_match_fine"]["_gt"]:
            for param in chain(
                map(
                    lambda x: x.parameters(),
                    [self.loftr_fine, self.fine_preprocess, self.fine_matching],
                )
            ):
                param.requires_grad = False
        
        self.extract_coarse_feats_mode = extract_coarse_feats_mode # Skip fine matching and use coarse mkpts to extract coarse fine features
        if self.extract_coarse_feats_mode:
            logger.warning("extract coarse features mode is open! skip fine matching!")

    def forward(self, data, **kwargs):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
                'extract_feature_mode' : used for neural_sfm no refinement ablation experiment skip fine matching and use coarse keypoint to extract features 
            }
        """
        # TODO: this should be removed in the future @zehong
        if self.loftr_coarse_pretrained:
            self.loftr_coarse.eval()

        # 1. local feature backbone
        with self.profiler.record_function("LoFTR/backbone"):
            data.update(
                {
                    "bs": data["image0"].size(0),
                    "hw0_i": data["image0"].shape[2:],
                    "hw1_i": data["image1"].shape[2:],
                }
            )
            if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
                feats = self.backbone(
                    torch.cat([data["image0"], data["image1"]], dim=0)
                )
                feats0, feats1 = _split_backbone_feats(feats, data["bs"])
            else:  # handle input of different shapes
                feats0, feats1 = map(self.backbone, [data["image0"], data["image1"]])

            feat_b_c0, feat_f0 = _extract_backbone_feats(
                feats0, self.config["loftr_backbone"]
            )
            feat_b_c1, feat_f1 = _extract_backbone_feats(
                feats1, self.config["loftr_backbone"]
            )
            data.update(
                {
                    "hw0_c": feat_b_c0.shape[2:],
                    "hw1_c": feat_b_c1.shape[2:],
                    "hw0_f": feat_f0.shape[2:],
                    "hw1_f": feat_f1.shape[2:],
                }
            )
            feat_c0_backbone = feat_b_c0.clone()
            feat_c1_backbone = feat_b_c1.clone()
            feat_f0_backbone = feat_f0.clone()
            feat_f1_backbone = feat_f1.clone()

        if "mkpts0_c" not in data:
            # 2. coarse-level loftr module
            with self.profiler.record_function("LoFTR/coarse-loftr"):
                # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
                feat_c0 = rearrange(
                    self.pos_encoding(feat_b_c0), "n c h w -> n (h w) c"
                )
                feat_c1 = rearrange(
                    self.pos_encoding(feat_b_c1), "n c h w -> n (h w) c"
                )

                # handle padding mask, for MegaDepth dataset
                mask_c0 = mask_c1 = None
                if "mask0" in data:
                    mask_c0, mask_c1 = (
                        data["mask0"].flatten(-2),
                        data["mask1"].flatten(-2),
                    )
                # NOTE: feat_c0 & feat_c1 are conv features residually modulated by LoFTR: x + sum_i(self_i + cross_i)
                feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
                # logger.info('Profiling LoFTR model...')
                # torch_speed_test(self.loftr_coarse, [feat_c0, feat_c1, mask_c0, mask_c1], model_name='loftr_coarse')

            with self.profiler.record_function("LoFTR/ssk-merge"):
                pass
                # TODO: concat along batch-dim if possible
                # TODO: Remove
                # feat_c0 = rearrange(feat_c0, 'n (h w) c -> n c h w', h=data['hw0_c'][0], w=data['hw0_c'][1])
                # feat_c1 = rearrange(feat_c1, 'n (h w) c -> n c h w', h=data['hw1_c'][0], w=data['hw1_c'][1])
                # feat_c0 = self.coarse_ssk_merge(feat_b_c0, feat_c0)
                # feat_c1 = self.coarse_ssk_merge(feat_b_c1, feat_c1)
                # feat_c0, feat_c1 = map(lambda x: rearrange(x, 'n c h w -> n (h w) c'), [feat_c0, feat_c1])

            with self.profiler.record_function("LoFTR/coarse-prior"):
                # option1: convolution (3x3-3x3-1x1-1x1) with detached loftr feature
                self.coarse_prior(feat_c0, feat_c1, data)

            # TODO: estimate dustin prototypes separately with feat_c0 & feat_c1 (AvgPool + MLP)
            # with self.profiler.record_function("LoFTR/coarse-prototype"):
            #     self.coarse_prototype(feat_c0, feat_c1, data)

            # 3. match coarse-level
            with self.profiler.record_function("LoFTR/coarse-matching"):
                self.coarse_matching(
                    feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1
                )

            if not self.config["loftr_match_fine"]["enable"]:
                data.update(
                    {"mkpts0_f": data["mkpts0_c"], "mkpts1_f": data["mkpts1_c"],}
                )
                return
        else:
            # Only fine with coarse provided
            # Convert coarse match to b_ids, i_ids, j_ids
            # NOTE: only allow bs == 1
            b_ids = torch.zeros(
                (data["mkpts0_c"].shape[0],), device=data["mkpts0_c"].device
            ).long()

            data['mkpts0_c'][:, 0] = torch.clip(data['mkpts0_c'][:, 0], min=0, max=data['hw0_i'][1] - 2)
            data['mkpts0_c'][:, 1] = torch.clip(data['mkpts0_c'][:, 1], min=0, max=data['hw0_i'][0] - 2)
            data['mkpts1_c'][:, 0] = torch.clip(data['mkpts1_c'][:, 0], min=0, max=data['hw1_i'][1] - 2)
            data['mkpts1_c'][:, 1] = torch.clip(data['mkpts1_c'][:, 1], min=0, max=data['hw1_i'][0] - 2)

            scale = data["hw0_i"][0] / data["hw0_c"][0]
            scale0 = (
                scale * data["scale0"][b_ids][:, [1, 0]] if "scale0" in data else scale
            )
            scale1 = (
                scale * data["scale1"][b_ids][:, [1, 0]] if "scale1" in data else scale
            )

            mkpts0_coarse_scaled = torch.round(data["mkpts0_c"] / scale0)
            mkpts1_coarse_scaled = torch.round(data["mkpts1_c"] / scale1)
            i_ids = (
                mkpts0_coarse_scaled[:, 1] * data["hw0_c"][1]
                + mkpts0_coarse_scaled[:, 0]
            ).long()
            j_ids = (
                mkpts1_coarse_scaled[:, 1] * data["hw1_c"][1]
                + mkpts1_coarse_scaled[:, 0]
            ).long()

            feat_c0, feat_c1 = None, None

            data.update(
                {"m_bids": b_ids, "b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}
            )

        if not self.extract_coarse_feats_mode:
            # 4. fine-level refinement
            with self.profiler.record_function("LoFTR/fine-refinement"):
                feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
                    feat_f0, feat_f1, feat_c0, feat_c1, data, feats0=feats0, feats1=feats1
                )
                feat_f0_raw, feat_f1_raw = feat_f0_unfold.clone(), feat_f1_unfold.clone()
                # at least one coarse level predicted
                if feat_f0_unfold.size(0) != 0 and self.enable_fine_loftr:
                    feat_f0_unfold, feat_f1_unfold = self.loftr_fine(
                        feat_f0_unfold, feat_f1_unfold
                    )

            # 5. match fine-level
            with self.profiler.record_function("LoFTR/fine-matching"):
                # TODO: add `cfg.FINE_MATCHING.ENABLE`
                self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

            # 6. (optional) fine-level rejection (with post loftr local feature)
            with self.profiler.record_function("LoFTR/fine-rejection"):
                feat_f0_rej, feat_f1_rej = (
                    (feat_f0_unfold, feat_f1_unfold)
                    if self.config["loftr_fine"]["rejector"]["post_loftr"]
                    else (feat_f0_raw, feat_f1_raw)
                )
                self.fine_rejector(feat_f0_rej, feat_f1_rej, data)

            # 7. (optional) Guided matching of existing detections
            with self.profiler.record_function("LoFTR/guided-matching"):
                self.guided_matching(data)

            with self.profiler.record_function("SfM pose refinement"):
                # data.update({"feats0":feats0, "feats1":feats1}) # backbone features ['coarse', 'fine']
                # data.update({"feat_c0": feat_c0, "feat_c1": feat_c1}) # coarse feature after loftr feature coarse
                # data.update({'feat_f0' : feat_f0, 'feat_f1' : feat_f1}) # fine feature backbone
                data.update(
                    {"feat_f0_unfold": feat_f0_unfold, "feat_f1_unfold": feat_f1_unfold}
                )
        else:
            data.update(
                {"mkpts0_f": data["mkpts0_c"], "mkpts1_f": data["mkpts1_c"],}
            )


            # self.pose_depth_refinement(data, fine_preprocess=self.fine_preprocess_unfold_none_grid, loftr_fine=self.loftr_fine)

        # Extract fine_feature (optional):
        if "extract_coarse_feature" in kwargs:
            if kwargs["extract_coarse_feature"]:
                # NOTE: Use mkpts0/1_f is not a bug, try to use their nearest coarse feature
                feat_coarse_b_0 = sample_feature_from_featuremap(
                    feat_c0_backbone,
                    data["mkpts0_f"],
                    imghw=data["scale0"].squeeze(0)
                    * torch.tensor(data["hw0_i"]).to(data["scale0"]),
                    sample_mode="nearest",
                )
                feat_coarse_b_1 = sample_feature_from_featuremap(
                    feat_c1_backbone,
                    data["mkpts1_f"],
                    imghw=data["scale1"].squeeze(0)
                    * torch.tensor(data["hw1_i"]).to(data["scale1"]),
                    sample_mode="nearest",
                )
                data.update(
                    {
                        "feat_coarse_b_0": feat_coarse_b_0,
                        "feat_coarse_b_1": feat_coarse_b_1,
                    }
                )
        if "extract_fine_feature" in kwargs:
            if kwargs["extract_fine_feature"]:
                feat_ext0 = sample_feature_from_featuremap(
                    feat_f0_backbone,
                    data["mkpts0_f"],
                    imghw=data["scale0"].squeeze(0)
                    * torch.tensor(data["hw0_i"]).to(data["scale0"]),
                )
                feat_ext1 = sample_feature_from_featuremap(
                    feat_f1_backbone,
                    data["mkpts1_f"],
                    imghw=data["scale1"].squeeze(0)
                    * torch.tensor(data["hw1_i"]).to(data["scale1"]),
                )
                data.update({"feat_ext0": feat_ext0, "feat_ext1": feat_ext1})

        # Wrap image1 and kpts to sample wrapped feature:
        if "homo_mat" in data:
            # Get warpped feature1
            h, w = data["hw1_i"]
            homo_sampled_normed = normalize_homography(
                data["homo_mat"], (h, w), (h, w)
            ).to(torch.float32)  # B*3*3
            # homo_warpped_image1 = transform.warp((pair_data['image1'].cpu().numpy() * 255).round().astype(np.int32)[0], homo_sampled)
            homo_warpped_image1 = homography_warp(
                data["image1"], torch.linalg.inv(homo_sampled_normed), (h, w)
            )

            image1_warpped_feat = self.backbone(homo_warpped_image1)
            image1_warpped_feat_b_c, image1_warpped_feat_f = _extract_backbone_feats(
                image1_warpped_feat, self.config["loftr_backbone"]
            )

            # Get warpped mkpts1
            norm_pixel_mat = normal_transform_pixel(h, w).to(data['mkpts1_f'].device)  # B*3*3 (1*3*3)
            mkpts1_normed = (
                norm_pixel_mat[0]
                @ (
                    torch.cat(
                        [data["mkpts1_f"].to(torch.float32), torch.ones((data["mkpts1_f"].shape[0], 1), device=data['mkpts1_f'].device)],
                        dim=-1,
                    )
                ).transpose(0, 1)
            )
            mkpts1_warpped = (
                norm_pixel_mat[0].inverse() @ homo_sampled_normed[0] @ mkpts1_normed
            ).transpose(
                0, 1
            )  # N*3
            mkpts1_warpped[:, :2] /= mkpts1_warpped[:, [2]]  # N*2
            mkpts1_warpped = mkpts1_warpped[:, :2]
            out_of_boundry_mask = (mkpts1_warpped[:,0] < 0) | (mkpts1_warpped[:,0] > (w - 1)) | (mkpts1_warpped[:,1] < 0) | (mkpts1_warpped[:, 1] > (h-1))

            # Sample kpts' features from feature map
            feat_coarse_b_1_warpped = sample_feature_from_featuremap(
                image1_warpped_feat_b_c,
                mkpts1_warpped,
                imghw=data["scale1"].squeeze(0)
                * torch.tensor(data["hw1_i"]).to(data["scale1"]),
                sample_mode="bilinear",
            )
            feat_f_1_warpped = sample_feature_from_featuremap(
                image1_warpped_feat_f,
                mkpts1_warpped,
                imghw=data["scale1"].squeeze(0)
                * torch.tensor(data["hw1_i"]).to(data["scale1"]),
            )

            # Use not warpped feature to substitute out of boundery feature:
            if "feat_coarse_b_1" not in data or "feat_ext1" not in data:
                raise NotImplementedError
            feat_coarse_b_1_warpped[out_of_boundry_mask, :] = data['feat_coarse_b_1'][out_of_boundry_mask, :]
            feat_f_1_warpped[out_of_boundry_mask] = data['feat_ext1'][out_of_boundry_mask, :]

            data.update(
                {
                    "feat_coarse_b_1_warpped": feat_coarse_b_1_warpped,
                    "feat_f_1_warpped": feat_f_1_warpped,
                }
            )

            # # Debug: plot match pair
            # from src.utils.plot_utils import make_matching_plot
            # import matplotlib.pyplot as plt
            # import numpy as np
            # original_image1 = (
            #     (data["image1"].cpu().numpy() * 255).round().astype(np.int32)[0]
            # )  # 1*H*W
            # homo_warpped_image1 = (
            #     (homo_warpped_image1.cpu().numpy() * 255).round().astype(np.int32)[0]
            # )  # 1*H*W
            # mkpts1 = data['mkpts1_f'][~out_of_boundry_mask].cpu().numpy()
            # mkpts1_warpped = mkpts1_warpped[~out_of_boundry_mask].cpu().numpy()
            # figure = make_matching_plot(
            #     original_image1[0],
            #     homo_warpped_image1[0],
            #     mkpts1,
            #     mkpts1_warpped,
            #     mkpts1,
            #     mkpts1_warpped,
            #     color=np.zeros((mkpts1.shape[0], 3)),
            #     text="",
            # )
            # plt.savefig("wrapped_match_pair.png")
            # plt.close()