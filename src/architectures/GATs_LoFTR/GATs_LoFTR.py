from itertools import chain

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from src.utils.profiler import PassThroughProfiler
from src.utils.misc import upper_config  # TODO: Remove the out of package import
from src.utils.torch_utils import torch_speed_test

from .backbone import (
    build_backbone,
    _extract_backbone_feats,
    _split_backbone_feats,
    _get_feat_dims,
)
from .utils.normalize import normalize_2d_keypoints, normalize_3d_keypoints
from .loftr_module import LocalFeatureTransformer
from .utils.position_encoding import PositionEncodingSine, KeypointEncoding
# from .utils.coarse_matching import CoarseMatching
# from .utils.fine_matching import FineMatching
# from .utils.selective_kernel import build_ssk_merge
# from .utils.guided_matching_fine import build_guided_matching

# from .two_view_refinement.pose_depth_refinement import PoseDepthRefinement


class GATs_LoFTR(nn.Module):
    def __init__(self, config, profiler=None, debug=False):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.debug = debug
        # self.enable_fine_loftr = self.config["loftr_fine"]["enable"]

        # Modules
        # Used to extract 2D query image feature
        self.backbone = build_backbone(self.config["loftr_backbone"])

        # For query image and 3D points
        self.dense_pos_encoding = PositionEncodingSine(
            self.config["loftr_coarse"]["d_model"],
            max_shape=self.config["loftr_coarse"]["pos_emb_shape"],
        )
        # NOTE: from Gats part
        self.kpt_3d_pos_encoding = KeypointEncoding(
            inp_dim=4,
            feature_dim=self.config['keypoints_encoding']["descriptor_dim"],
            layers=self.config['keypoints_encoding']["keypoints_encoder"],
        )

        self.loftr_coarse = LocalFeatureTransformer(self.config["loftr_coarse"])

        # self.coarse_matching = CoarseMatching(
        #     self.config["loftr_match_coarse"],
        #     self.config["loftr_match_fine"]["detector"],
        #     profiler=self.profiler,
        # )

        # self.fine_preprocess = FinePreprocess(
        #     self.config["loftr_fine"],
        #     self.config["loftr_coarse"]["d_model"],
        #     cf_res=self.config["loftr_backbone"]["resolution"],
        #     feat_ids=self.config["loftr_backbone"]["resnetfpn"]["output_layers"],
        #     feat_dims=_get_feat_dims(self.config["loftr_backbone"]),
        # )

        # self.loftr_fine = LocalFeatureTransformer(self.config["loftr_fine"])
        # self.fine_matching = FineMatching(
        #     self.config["loftr_match_fine"], _full_cfg=upper_config(self.config)
        # )

        # Optional Modules
        # self.coarse_prior = build_coarse_prior(self.config["loftr_coarse"])
        # self.fine_rejector = build_rejector(self.config["loftr_fine"])
        # self.coarse_ssk_merge = build_ssk_merge(self.config["loftr_coarse"])
        # self.guided_matching = build_guided_matching(
        #     self.config["loftr_guided_matching"]
        # )

        # self.pose_depth_refinement = PoseDepthRefinement(config['loftr_sfm'])

        # FIXME: Change to backbone
        # fixed pretrained coarse weights
        # self.loftr_coarse_pretrained = self.config["loftr_coarse"]["pretrained"]
        # if self.loftr_coarse_pretrained is not None:
        #     ckpt = torch.load(self.loftr_coarse_pretrained, "cpu")["state_dict"]
        #     for k in list(ckpt.keys()):
        #         if "loftr_coarse" in k:
        #             newk = k[k.find("loftr_coarse") + len("loftr_coarse") + 1 :]
        #             ckpt[newk] = ckpt[k]
        #         if "coarse_matching" in k:
        #             newk = k[k.find("coarse_matching") + len("coarse_matching") + 1 :]
        #             self.coarse_matching.load_state_dict({newk: ckpt[k]})
        #             self.coarse_matching.requires_grad_(False)
        #         ckpt.pop(k)
        #     self.loftr_coarse.load_state_dict(ckpt)
        #     for param in self.loftr_coarse.parameters():
        #         param.requires_grad = False

        # # Disable grads when use gt mode (for convenience, inference without backprop, but better to disable)
        # if self.config["loftr_match_coarse"]["_gt"]:
        #     for param in self.loftr_coarse.parameters():
        #         param.requires_grad = False
        # if self.config["loftr_match_fine"]["_gt"]:
        #     for param in chain(
        #         map(
        #             lambda x: x.parameters(),
        #             [self.loftr_fine, self.fine_preprocess, self.fine_matching],
        #         )
        #     ):
        #         param.requires_grad = False

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                keypoints3d: [N, n1, 3]
                descriptors3d_db: [N, dim, n1]
                descriptors2d_db: [N, dim, n1 * num_leaf]
                scores3d_db: [N, n1, 1]
                scores2d_db: [N, n1 * num_leaf, 1]

                query_image: (N, 1, H, W)
                query_image_scale: (N, 2)
                query_image_mask(optional): (N, H, W)
            }
        """
        # TODO: this should be removed in the future @zehong
        # if self.loftr_coarse_pretrained:
        #     self.loftr_coarse.eval()

        # 1. local feature backbone
        with self.profiler.record_function("LoFTR/backbone"):
            data.update(
                {
                    "bs": data["query_image"].size(0),
                    "q_hw_i": data["query_image"].shape[2:],
                }
            )
            # if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
            #     feats = self.backbone(
            #         torch.cat([data["image0"], data["image1"]], dim=0)
            #     )
            #     feats0, feats1 = _split_backbone_feats(feats, data["bs"])
            # else:  # handle input of different shapes
            #     feats0, feats1 = map(self.backbone, [data["image0"], data["image1"]])
            query_feature_map = self.backbone(data['query_image'])

            query_feat_b_c, query_feat_f = _extract_backbone_feats(
                query_feature_map, self.config["loftr_backbone"]
            )
            data.update(
                {
                    "q_hw_c": query_feat_b_c.shape[2:],
                    "q_hw_f": query_feat_f.shape[2:],
                }
            )

        kpts3d = normalize_3d_keypoints(data['keypoints3d'])

        # 2. coarse-level loftr module
        with self.profiler.record_function("LoFTR/coarse-loftr"):
            # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
            query_feat_c = rearrange(
                self.dense_pos_encoding(query_feat_b_c), "n c h w -> n (h w) c"
            )

            desc3d_db = self.kpt_3d_pos_encoding(kpts3d, data['scores3d_db'],data['descriptors3d_db'])
            desc2d_db = data['descriptor2d_db']

            # handle padding mask, for MegaDepth dataset
            query_mask = None
            if "query_image_mask" in data:
                query_mask = data['query_image_mask'].flatten(-2)
            # NOTE: feat_c0 & feat_c1 are conv features residually modulated by LoFTR: x + sum_i(self_i + cross_i)
            desc3d_db, query_feat_c = self.loftr_coarse(desc3d_db, desc2d_db, query_feat_c, query_mask=query_mask)
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

        # with self.profiler.record_function("LoFTR/coarse-prior"):
        #     # option1: convolution (3x3-3x3-1x1-1x1) with detached loftr feature
        #     self.coarse_prior(feat_c0, feat_c1, data)

        # TODO: estimate dustin prototypes separately with feat_c0 & feat_c1 (AvgPool + MLP)
        # with self.profiler.record_function("LoFTR/coarse-prototype"):
        #     self.coarse_prototype(feat_c0, feat_c1, data)

        # 3. match coarse-level
        # FIXME: not finish yet !
        with self.profiler.record_function("LoFTR/coarse-matching"):
            self.coarse_matching(
                feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1
            )

        if not self.config["loftr_match_fine"]["enable"]:
            data.update(
                {"mkpts0_f": data["mkpts0_c"], "mkpts1_f": data["mkpts1_c"],}
            )
            return


        # Only fine with coarse provided
        # Convert coarse match to b_ids, i_ids, j_ids
        # NOTE: only allow bs == 1
        b_ids = torch.zeros(
            (data["mkpts0_c"].shape[0],), device=data["mkpts0_c"].device
        ).long()

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

        # # Debug
        # mkpts0_c = torch.stack([i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]], dim=1) * scale0
        # mkpts1_c = torch.stack([j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]], dim=1) * scale1
        # print(torch.mean(mkpts0_c - data['mkpts0_c']))
        # print(torch.mean(mkpts1_c - data['mkpts1_c']))

        feat_c0, feat_c1 = None, None

        data.update(
            {"m_bids": b_ids, "b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}
        )

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

        # Pose regression
        # TODO: remove to a independent function in future
        with self.profiler.record_function("SfM pose refinement"):
            # data.update({"feats0":feats0, "feats1":feats1}) # backbone features ['coarse', 'fine']
            # data.update({"feat_c0": feat_c0, "feat_c1": feat_c1}) # coarse feature after loftr feature coarse
            # data.update({'feat_f0' : feat_f0, 'feat_f1' : feat_f1}) # fine feature backbone
            data.update(
                {"feat_f0_unfold": feat_f0_unfold, "feat_f1_unfold": feat_f1_unfold}
            )
            # self.pose_depth_refinement(data, fine_preprocess=self.fine_preprocess_unfold_none_grid, loftr_fine=self.loftr_fine)
