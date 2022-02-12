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
from .utils.normalize import normalize_3d_keypoints
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.position_encoding import PositionEncodingSine, KeypointEncoding, KeypointEncoding_linear, PositionEncodingSine3D
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class GATs_LoFTR(nn.Module):
    def __init__(self, config, profiler=None, debug=False):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.debug = debug

        # Modules
        # Used to extract 2D query image feature
        self.backbone = build_backbone(self.config["loftr_backbone"])

        # For query image and 3D points
        if self.config["positional_encoding"]["enable"]:
            self.dense_pos_encoding = PositionEncodingSine(
                self.config["loftr_coarse"]["d_model"],
                max_shape=self.config["positional_encoding"]["pos_emb_shape"],
            )
        else:
            self.dense_pos_encoding = None

        if self.config["keypoints_encoding"]["enable"]:
            # NOTE: from Gats part
            if config['keypoints_encoding']['type'] == "mlp_cov1d":
                encoding_func = KeypointEncoding
            elif config['keypoints_encoding']['type'] == "mlp_linear": 
                encoding_func = KeypointEncoding_linear
            elif config['keypoints_encoding']['type'] == "sine": 
                encoding_func = PositionEncodingSine3D
            else:
                raise NotImplementedError

            self.kpt_3d_pos_encoding = encoding_func(
                inp_dim=3,
                feature_dim=self.config["keypoints_encoding"]["descriptor_dim"],
                layers=self.config["keypoints_encoding"]["keypoints_encoder"],
                norm_method=self.config["keypoints_encoding"]["norm_method"],
            )
        else:
            self.kpt_3d_pos_encoding = None

        self.loftr_coarse = LocalFeatureTransformer(self.config["loftr_coarse"])

        self.coarse_matching = CoarseMatching(
            self.config["coarse_matching"], profiler=self.profiler,
        )

        self.fine_preprocess = FinePreprocess(
            self.config["loftr_fine"],
            self.config["loftr_coarse"]["d_model"],
            cf_res=self.config["loftr_backbone"]["resolution"],
            feat_ids=self.config["loftr_backbone"]["resnetfpn"]["output_layers"],
            feat_dims=_get_feat_dims(self.config["loftr_backbone"]),
        )

        self.loftr_fine = LocalFeatureTransformer(self.config["loftr_fine"])
        self.fine_matching = FineMatching(self.config["fine_matching"])

        self.loftr_backbone_pretrained = self.config["loftr_backbone"]["pretrained"]
        if self.loftr_backbone_pretrained is not None:
            logger.info(
                f"Load pretrained backbone from {self.loftr_backbone_pretrained}"
            )
            ckpt = torch.load(self.loftr_backbone_pretrained, "cpu")["state_dict"]
            for k in list(ckpt.keys()):
                if "backbone" in k:
                    newk = k[k.find("backbone") + len("backbone") + 1 :]
                    ckpt[newk] = ckpt[k]
                ckpt.pop(k)
            self.backbone.load_state_dict(ckpt)

            if self.config["loftr_backbone"]["pretrained_fix"]:
                for param in self.backbone.parameters():
                    param.requires_grad = False

    def forward(self, data, return_fine_unfold_feat=False):
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
        if (
            self.loftr_backbone_pretrained
            and self.config["loftr_backbone"]["pretrained_fix"]
        ):
            self.backbone.eval()

        # 1. local feature backbone
        with self.profiler.record_function("LoFTR/backbone"):
            data.update(
                {
                    "bs": data["query_image"].size(0),
                    "q_hw_i": data["query_image"].shape[2:],
                }
            )

            query_feature_map = self.backbone(data["query_image"])

            query_feat_b_c, query_feat_f = _extract_backbone_feats(
                query_feature_map, self.config["loftr_backbone"]
            )
            data.update(
                {"q_hw_c": query_feat_b_c.shape[2:], "q_hw_f": query_feat_f.shape[2:],}
            )

        if self.config["use_fine_backbone_as_coarse"]:
            # Down sample fine feature
            query_feat_b_c = F.interpolate(
                query_feat_f,
                size=query_feat_b_c.shape[2:],
                mode=self.config["interpol_type"],
                align_corners=False
            )

        # 2. coarse-level loftr module
        with self.profiler.record_function("LoFTR/coarse-loftr"):
            # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
            query_feat_c = rearrange(
                self.dense_pos_encoding(query_feat_b_c)
                if self.dense_pos_encoding is not None
                else query_feat_b_c,
                "n c h w -> n (h w) c",
            )

            kpts3d = normalize_3d_keypoints(data["keypoints3d"])
            desc3d_db = (
                self.kpt_3d_pos_encoding(
                    kpts3d, data["descriptors3d_db"] if "descriptors3d_coarse_db" not in data else data['descriptors3d_coarse_db']
                )
                if self.kpt_3d_pos_encoding is not None
                else data["descriptors3d_db"] if "descriptors3d_coarse_db" not in data else data['descriptors3d_coarse_db']
            )
            desc2d_db = data["descriptors2d_db"] if 'descriptors2d_coarse_db' not in data else data['descriptors2d_coarse_db']

            query_mask = None
            if "query_image_mask" in data:
                query_mask = data["query_image_mask"].flatten(-2)
            # NOTE: feat_c0 & feat_c1 are conv features residually modulated by LoFTR: x + sum_i(self_i + cross_i)
            desc3d_db, query_feat_c = self.loftr_coarse(
                desc3d_db, desc2d_db, query_feat_c, data, query_mask=query_mask, keypoints3D=kpts3d
            )
            # logger.info('Profiling LoFTR model...')
            # torch_speed_test(self.loftr_coarse, [feat_c0, feat_c1, mask_c0, mask_c1], model_name='loftr_coarse')

        # 3. match coarse-level
        with self.profiler.record_function("LoFTR/coarse-matching"):
            self.coarse_matching(desc3d_db, query_feat_c, data, mask_query=query_mask)

        if not self.config["fine_matching"]["enable"]:
            data.update(
                {
                    "mkpts_3d_db": data["mkpts_3d_db"],
                    "mkpts_query_f": data["mkpts_query_c"],
                }
            )
            return

        # 4. fine-level refinement
        with self.profiler.record_function("LoFTR/fine-refinement"):
            (
                desc3d_db_selected,
                desc2d_db_selected,
                query_feat_f_unfolded,
            ) = self.fine_preprocess(
                data, data["descriptors3d_db"], data["descriptors2d_db"], query_feat_f, feat_3D_c=desc3d_db, feat_query_c=query_feat_c
            )
            # at least one coarse level predicted
            if (
                query_feat_f_unfolded.size(0) != 0
                and self.config["loftr_fine"]["enable"]
            ):
                desc3d_db_selected, query_feat_f_unfolded = self.loftr_fine(
                    desc3d_db_selected, desc2d_db_selected, query_feat_f_unfolded, data
                )
            else:
                desc3d_db_selected = torch.einsum("bdn->bnd", desc3d_db_selected)  # [N, L, C]

        if return_fine_unfold_feat:
            data.update({'desc3d_db_selected': desc3d_db_selected, 'query_feat_f_unfolded': query_feat_f_unfolded})

        # 5. match fine-level
        with self.profiler.record_function("LoFTR/fine-matching"):
            self.fine_matching(desc3d_db_selected, query_feat_f_unfolded, data)