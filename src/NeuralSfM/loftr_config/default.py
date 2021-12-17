from yacs.config import CfgNode as CN

_CN = CN()
_CN.PIPELINE = "spg"  # options: spg, loftr, loctr, mnn, loftr_gm, loftr_sfm

##############  Extractor ##############
# SuperPoint config
_CN.SUPERPOINT = CN()
_CN.SUPERPOINT.NMS_RADIUS = 4
_CN.SUPERPOINT.REMOVE_BORDERS = 4
_CN.SUPERPOINT.KEYPOINT_THRESHOLD = 0.005
_CN.SUPERPOINT.MAX_KEYPOINTS = 400  # 400 KPTS FOR INDOOR SCENE
_CN.SUPERPOINT.DESCRIPTOR_DIM = 256
# SVCNN config
_CN.SVCNN = CN()
_CN.SVCNN.NMS_RADIUS = 4
_CN.SVCNN.REMOVE_BORDERS = 4
_CN.SVCNN.KEYPOINT_THRESHOLD = 0.005
_CN.SVCNN.MAX_KEYPOINTS = 1024
_CN.SVCNN.DESCRIPTOR_DIM = 128
_CN.SVCNN.SCALES= [1]
_CN.SVCNN.DO_QUADRATIC_REFINEMENT=0
_CN.SVCNN.REFINEMENT_RADIUS=0
_CN.SVCNN.HARRIS_RADIUS=0
_CN.SVCNN.VERSION="v9"

# Disk config
_CN.DISK = CN()
_CN.DISK.MAX_KEYPOINTS=1024
_CN.DISK.WINDOW_SIZE_RNG=8
_CN.DISK.WINDOW_SIZE_NMS=5 #requirement: window_size_nms//2!=0
_CN.DISK.CUT_OFF=0.
_CN.DISK.KIND="nms"  #choice:["nms","rng"]
_CN.DISK.DESCRIPTOR_DIM=128
_CN.DISK.KERNEL_SIZE=5  #kernel size for u-net
_CN.DISK.KEYPOINT_THRESHOLD=0.005 #same with SVCNN and superpoint

# R2D2 config
_CN.R_TWO_D_TWO=CN()
_CN.R_TWO_D_TWO.MAX_KEYPOINTS=1024
_CN.R_TWO_D_TWO.DESCRIPTOR_DIM=128
_CN.R_TWO_D_TWO.VERSION="WAF_N16"  #choice:["WAF_N16","WASF_N8_big","WASF_N16"]
_CN.R_TWO_D_TWO.REL_THR=0.7
_CN.R_TWO_D_TWO.REP_THR=0.7
_CN.R_TWO_D_TWO.KEYPOINT_THRESHOLD=0.005
#R2D2 multi-scale configs
_CN.R_TWO_D_TWO.MIN_SCALE=1
_CN.R_TWO_D_TWO.MAX_SCALE=1
_CN.R_TWO_D_TWO.SCALE_F=2**0.25

# D2 config
_CN.D_TWO_NET=CN()
_CN.D_TWO_NET.MAX_KEYPOINTS=1024
_CN.D_TWO_NET.DESCRIPTOR_DIM=512
_CN.D_TWO_NET.VERSION="ots"  #["ots","tf","tf_no_phototourism"]
_CN.D_TWO_NET.SCALES=[1]
_CN.D_TWO_NET.KEYPOINT_THRESHOLD=0.005

##############  ↓  SPG Pipeline  ↓  ##############
# 1. SuperGlue config
_CN.SUPERGLUE = CN()
_CN.SUPERGLUE.DETECTOR_TYPE= "SuperPoint"  #{'SuperPoint','SVCNN','Disk'}
_CN.SUPERGLUE.WEIGHTS = 'indoor'  # only useful when loading pretrained model
_CN.SUPERGLUE.SINKHORN_ITERATIONS = 20
_CN.SUPERGLUE.MATCH_THRESHOLD = 0.2
_CN.SUPERGLUE.DESCRIPTOR_DIM = 256
_CN.SUPERGLUE.KEYPOINT_ENCODER = [32, 64, 128, 256]
_CN.SUPERGLUE.GNN_LAYERS = ['self', 'cross'] * 9
_CN.SUPERGLUE.ATTENTION = 'vanilla'  # {'vanilla', 'linear_transformer'}
_CN.SUPERGLUE.IGNORE_REPROJ = 14  # THE IGNORED REPROJECTION ERROR VALUE' - SET AS SINGLETON
_CN.SUPERGLUE.DEPTH_THRESH = 0.1  # Absolute tolerance of depth consistency check (in meters)
_CN.SUPERGLUE.RELATIVE_DEPTH_THRESH = 0.0  # Relative tolerance of depth consistency check
_CN.SUPERGLUE.POS_REPROJ_THRESH = 5  # THE MAX REPROJECTION ERROR THRESHOLD FOR POSITIVE MATCHING PAIRS
_CN.SUPERGLUE.NEG_REPROJ_THRESH = 15  # THE MIN REPROJEC
_CN.SUPERGLUE.LOAD_ORIGINAL_GLUE_WEIGHT = False
_CN.SUPERGLUE.WEIGHT_INIT = False
_CN.SUPERGLUE.INIT_BIN_SCORE = 1.0  # The author suggests to use a smaller bin_score like 0.0
_CN.SUPERGLUE.MATCHING_TYPE = 'sinkhorn'  #{'sinkhorn','dual_softmax'}
_CN.SUPERGLUE.ENABLE_ATTENTION = True

# 2. Superglue Loss
_CN.SUPERGLUE_LOSS = CN()
_CN.SUPERGLUE_LOSS.REDUCTION = 'mean'
_CN.SUPERGLUE_LOSS.TYPE = 'nll'  # ['nll', 'focal']
# focal loss
_CN.SUPERGLUE_LOSS.FOCAL_ALPHA = 0.25
_CN.SUPERGLUE_LOSS.FOCAL_GAMMA = 2.0
# loss weights for positive and negative terms.
# if None, all 3 terms in the loss with all be assigned a weight of 1/3.
# When using both focal loss and manually assigning weights to pos & neg terms
# both the FOCAL_GAMMA and XXX_WEIGHT shold be considered.
# The learning rate might need to be re-tuned
_CN.SUPERGLUE_LOSS.POS_WEIGHT = None  # The weight assigned to the positive loss term
_CN.SUPERGLUE_LOSS.NEG_WEIGHT = None  # The weight assigned to the negative loss terms,
# two parts of the negative loss will be assigned 0.5*NEG_WEIGHT each

##############  ↓  LoFTR Pipeline  ↓  ##############
# 1. LoFTR-backbone (local feature CNN) config
_CN.LOFTR_BACKBONE = CN()
_CN.LOFTR_BACKBONE.TYPE = 'ResNetFPN'  # ['ResNetFPN', 'Hybrid', 'Res2']
_CN.LOFTR_BACKBONE.RESOLUTION = (8, 2)  # [(8, 2), (16, 4), (16, 2)]
# ResNetFPN config
_CN.LOFTR_BACKBONE.RESNETFPN = CN()
_CN.LOFTR_BACKBONE.RESNETFPN.BLOCK_TYPE = 'BasicBlock'
_CN.LOFTR_BACKBONE.RESNETFPN.INITIAL_DIM = 64
_CN.LOFTR_BACKBONE.RESNETFPN.BLOCK_DIMS = [64, 96, 128]  # s1, s2, s3
_CN.LOFTR_BACKBONE.RESNETFPN.OUTPUT_LAYERS = [3, 1]
# Hybrid config
_CN.LOFTR_BACKBONE.HYBRID = CN()
_CN.LOFTR_BACKBONE.HYBRID.IMG_SIZE = (480, 640)  # FIXME: REPEAT (XXX_IMG_RESIZE)
_CN.LOFTR_BACKBONE.HYBRID.IN_CHANS = 1  # [1, 3]
_CN.LOFTR_BACKBONE.HYBRID.PATCH_STRIDE = 16  # the patch size is determined by RESOLUTION
_CN.LOFTR_BACKBONE.HYBRID.DEPTH = 12  # depth of ViT
_CN.LOFTR_BACKBONE.HYBRID.PRETRAINED = True
_CN.LOFTR_BACKBONE.HYBRID.OUTPUT_DIMS = [128, 64]
_CN.LOFTR_BACKBONE.HYBRID.ATTN_TYPE = 'full'  # ['full', 'linear']
_CN.LOFTR_BACKBONE.HYBRID.ATTN_CFG = CN()
_CN.LOFTR_BACKBONE.HYBRID.ATTN_CFG.KERNEL_FN = 'Favor'
_CN.LOFTR_BACKBONE.HYBRID.ATTN_CFG.REDRAW_INTERVAL = 1
_CN.LOFTR_BACKBONE.HYBRID.ATTN_CFG.D_KERNEL = 192 // 3
# Res2 config
_CN.LOFTR_BACKBONE.RES2 = CN()
_CN.LOFTR_BACKBONE.RES2.COARSE_PRETRAINED = None
_CN.LOFTR_BACKBONE.RES2.COARSE = CN()
_CN.LOFTR_BACKBONE.RES2.COARSE.BLOCK_TYPE = 'BasicBlock'
_CN.LOFTR_BACKBONE.RES2.COARSE.INITIAL_DIM = 64
_CN.LOFTR_BACKBONE.RES2.COARSE.BLOCK_DIMS = [64, 96, 128]
_CN.LOFTR_BACKBONE.RES2.FINE = CN()
_CN.LOFTR_BACKBONE.RES2.FINE.BLOCK_TYPE = 'BasicBlock'
_CN.LOFTR_BACKBONE.RES2.FINE.INITIAL_DIM = 128
_CN.LOFTR_BACKBONE.RES2.FINE.BLOCK_DIMS = [128]

# 2. LoFTR-coarse module config
_CN.LOFTR_COARSE = CN()
_CN.LOFTR_COARSE.TYPE = 'LoFTR'  # ['LoFTR', 'Pre-LN', 'Post-LN', 'Rezero']
_CN.LOFTR_COARSE.D_MODEL = 128
_CN.LOFTR_COARSE.D_FFN = 128  # dim of feed-forward network
_CN.LOFTR_COARSE.NHEAD = 8
_CN.LOFTR_COARSE.LAYER_NAMES = ['self', 'cross'] * 8
_CN.LOFTR_COARSE.DROPOUT = 0.
_CN.LOFTR_COARSE.ATTENTION = 'linear' # list of str. Options: [linear, full, msd]
# The length should be equal to len(LAYER_NAMES) or 1.

_CN.LOFTR_COARSE.KERNEL_FN = 'elu + 1'  # ['elu + 1', 'Favor', 'GeneralizedRandomFeatures']
_CN.LOFTR_COARSE.D_KERNEL = 128 // 8
_CN.LOFTR_COARSE.REDRAW_INTERVAL = 2  # for perfomer
_CN.LOFTR_COARSE.REZERO = None  # for LoFTREncoderLayer only
_CN.LOFTR_COARSE.FINAL_PROJ = False
_CN.LOFTR_COARSE.POS_EMB_SHAPE = (256, 256)  # Shape of the positional embedding (should be bigger than max(img-size) // 8)
_CN.LOFTR_COARSE.MSD_N_POINTS = 8  # For multi-scale deformable attention

# optional - coarse prior estimation
_CN.LOFTR_COARSE.PRIOR = CN()
_CN.LOFTR_COARSE.PRIOR.ENABLE = False
_CN.LOFTR_COARSE.PRIOR.DETACH = True
# optional - coarse pretrained ckpt
_CN.LOFTR_COARSE.PRETRAINED = None
# optional - coarse spatial selective kerner merge
_CN.LOFTR_COARSE.SSK = CN()
_CN.LOFTR_COARSE.SSK.ENABLE = False
_CN.LOFTR_COARSE.SSK.D_ATTN = 32
_CN.LOFTR_COARSE.SSK.RESIDUAL = False

# 3. Coarse-Matching config
_CN.LOFTR_MATCH_COARSE = CN()
_CN.LOFTR_MATCH_COARSE.FEAT_NORM_METHOD = 'sqrt_feat_dim'  # ['sqrt_feat_dim', None, 'temperature']
_CN.LOFTR_MATCH_COARSE.TYPE = 'sinkhorn'  # ['sinkhorn', 'dual-softmax']
_CN.LOFTR_MATCH_COARSE.THR = 0.2  # tune when testing
_CN.LOFTR_MATCH_COARSE.BORDER_RM = 2
_CN.LOFTR_MATCH_COARSE.SPG_SPVS = False
_CN.LOFTR_MATCH_COARSE.SAVE_COARSE_ALL_MATCHES = False  # save coarse matches and spp keypoints, use for debug and visualize in LoFTR-spp
# sinkhorn config
_CN.LOFTR_MATCH_COARSE.SKH = CN()
_CN.LOFTR_MATCH_COARSE.SKH.FP16 = False  # For MegaDepth 640x640 input, MEM: 10.646GB v.s. 10.8GB
_CN.LOFTR_MATCH_COARSE.SKH.ITERS = 20
_CN.LOFTR_MATCH_COARSE.SKH.PARTIAL_IMPL = 'dustbin'  # ['dustbin', 'prototype', 'unbalanced']
_CN.LOFTR_MATCH_COARSE.SKH.INIT_BIN_SCORE = 1.0
_CN.LOFTR_MATCH_COARSE.SKH.PREFILTER = False
_CN.LOFTR_MATCH_COARSE.SKH.D_MODEL = 128  # FIXME: REPEAT with D_MODEL
_CN.LOFTR_MATCH_COARSE.SKH.PROTOTYPE_IMPL = 'learned'  # ['learned', 'mean', ...]
_CN.LOFTR_MATCH_COARSE.SKH.PREFILTER = False  # tune when testing
_CN.LOFTR_MATCH_COARSE.SKH.WITH_PRIOR = False  # with estimated prior
_CN.LOFTR_MATCH_COARSE.SKH.LINEAR = CN()
_CN.LOFTR_MATCH_COARSE.SKH.LINEAR.ENABLE = False
_CN.LOFTR_MATCH_COARSE.SKH.LINEAR.MAPPING = 'favor'  # ['favor', 'gauss', 'rff]
_CN.LOFTR_MATCH_COARSE.SKH.LINEAR.MAPPING_DIM = 4096
# dual-softmax config
_CN.LOFTR_MATCH_COARSE.DUAL_SOFTMAX = CN()
_CN.LOFTR_MATCH_COARSE.DUAL_SOFTMAX.TEMPERATURE = 0.1

# Guided Matching (used by both loftr & loftr_gm)
_CN.LOFTR_GUIDED_MATCHING = CN()
_CN.LOFTR_GUIDED_MATCHING.ENABLE = False
_CN.LOFTR_GUIDED_MATCHING.WINDOW_SIZE = None
_CN.LOFTR_GUIDED_MATCHING.MULTI_KPTS_IN_ONE = 'top-1'
_CN.LOFTR_GUIDED_MATCHING.NO_KPT_IN_ONE_WIN = 'discard'
_CN.LOFTR_GUIDED_MATCHING.KEEP_REFINED_PTS = False

# training
_CN.LOFTR_MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.4  # save GPU memory
_CN.LOFTR_MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # avoid deadlock; better convergence

# 4. LoFTR-fine module config
_CN.LOFTR_FINE = CN()
_CN.LOFTR_FINE.ENABLE = True
_CN.LOFTR_FINE.TYPE = 'LoFTR'  # ['LoFTR', 'Pre-LN', 'Post-LN', 'Rezero']
_CN.LOFTR_FINE.D_MODEL = 64
_CN.LOFTR_FINE.D_FFN = 64  # dim of feed-forward network
_CN.LOFTR_FINE.NHEAD = 8
_CN.LOFTR_FINE.LAYER_NAMES = ['self', 'cross'] * 2
_CN.LOFTR_FINE.DROPOUT = 0.
_CN.LOFTR_FINE.ATTENTION = 'linear'
_CN.LOFTR_FINE.KERNEL_FN = 'elu + 1'  # ['elu + 1', 'Favor', 'GeneralizedRandomFeatures']
_CN.LOFTR_FINE.D_KERNEL = 64 // 8
_CN.LOFTR_FINE.REDRAW_INTERVAL = 2  # for perfomer
_CN.LOFTR_FINE.MSD_N_POINTS = None  # For multi-scale deformable attention

_CN.LOFTR_FINE.WINDOW_SIZE = 5  # window_size in fine_level, must be odd
# optional - concat coarse-level loftr feature as context
_CN.LOFTR_FINE.CONCAT_COARSE_FEAT = False
_CN.LOFTR_FINE.CONCAT_COARSE_FEAT_TYPE = 'nearest'  # ['nearest', 'bilinear']
_CN.LOFTR_FINE.REZERO = None
_CN.LOFTR_FINE.FINAL_PROJ = False
# optional - concat coarse-level loftr feature as context
_CN.LOFTR_FINE.CONCAT_COARSE_FEAT = False
_CN.LOFTR_FINE.COARSE_LAYER_NORM = False  # the coarse-loftr feature has large value scale and variance
# optional - multi-scale feature processing
_CN.LOFTR_FINE.MS_FEAT = False  # whether use multi-scale feature fusion or not (only work when len(OUTPUT_LAYERS)==2)
_CN.LOFTR_FINE.MS_FEAT_TYPE = 'PROJ_MERGE'  # ['PROJ_MERGE', 'CAT_CONV']
_CN.LOFTR_FINE.D_MS_FEAT = 128  # proj to a higher dim -> add -> proj to D_MODEL

# optional - fine rejection
_CN.LOFTR_FINE.REJECTOR = CN()
_CN.LOFTR_FINE.REJECTOR.ENABLE = False
_CN.LOFTR_FINE.REJECTOR.THR = 0.5
_CN.LOFTR_FINE.REJECTOR.POST_LOFTR = False  # use fine-level feature after loftr, otherwise use raw feature from pyramid
# TODO: Concat coarse-level loftr feature as context for rejection.

# 5. Fine-Matching config
_CN.LOFTR_MATCH_FINE = CN()
_CN.LOFTR_MATCH_FINE.ENABLE = True
_CN.LOFTR_MATCH_FINE.TYPE = 's2d'
_CN.LOFTR_MATCH_FINE.DETECTOR = 'OnGrid'    # [OnGrid, SuperPoint, SuperPointEC,SuperPoint and grid]
_CN.LOFTR_MATCH_FINE.WINDOW_SIZE = 5  # FIXME: REPEAT
_CN.LOFTR_MATCH_FINE.FEATS_HDF5 = False  # else a path to features.h5
_CN.LOFTR_MATCH_FINE.SAVE_KEYPOINTS= False
# optional: SuperPoint config
_CN.LOFTR_MATCH_FINE.SPP = CN()
_CN.LOFTR_MATCH_FINE.SPP.KEYPOINT_THRESHOLD = 0.005
_CN.LOFTR_MATCH_FINE.SPP.MAX_KEYPOINTS = -1
_CN.LOFTR_MATCH_FINE.SPP.NMS_RADIUS = 4
# optional: SuperPointEC config
_CN.LOFTR_MATCH_FINE.SPPEC = CN()
_CN.LOFTR_MATCH_FINE.SPPEC.ALIGN_CENTER_WITH_RESNET = True  # vgg&resnet use different strategy to downsample
_CN.LOFTR_MATCH_FINE.SPPEC.VERSION = 'v1'  # [v1, v2], check py file for more details
# optional: SIFT config
_CN.LOFTR_MATCH_FINE.SIFT = CN()
_CN.LOFTR_MATCH_FINE.SIFT.N_FEATURES = 0
_CN.LOFTR_MATCH_FINE.SIFT.CONTRAST_THRESHOLD = 0.04  # -10000 for low_thr version
_CN.LOFTR_MATCH_FINE.SIFT.EDGE_THRESHOLD = 10.  # -10000 for low_thr version

# s2d variants
_CN.LOFTR_MATCH_FINE.S2D = CN()
_CN.LOFTR_MATCH_FINE.S2D.TYPE = 'heatmap'  # ['hetamap', 'regress']
_CN.LOFTR_MATCH_FINE.S2D.REGRESS = CN()
_CN.LOFTR_MATCH_FINE.S2D.REGRESS.TYPE = 'correlation'  # ['correlation', 'diff']
_CN.LOFTR_MATCH_FINE.S2D.REGRESS.CLS = False  # classification at the same time
_CN.LOFTR_MATCH_FINE.S2D.REGRESS.D = 64  # dim of MLP
_CN.LOFTR_MATCH_FINE.S2D.REGRESS.NORM = 'feat_dim'  # ['feat_dim', 'l2']

# GT Visualization
_CN.LOFTR_MATCH_COARSE._GT = False
_CN.LOFTR_MATCH_FINE._GT = False
_CN.LOFTR_MATCH_FINE._GT_TRUNC = False  # `expec_f_gt` might exceed window, truncate it to window border
_CN.LOFTR_MATCH_FINE._GT_NOISE = 0.  # level of noise added to `expec_f_gt`: U(-2, 2) * _GT_NOISE
_CN.LOFTR_MATCH_COARSE._N_RAND_SAMPLES = None  # used to determine the max possible image size

# 6. LoFTR loss
_CN.LOFTR_LOSS = CN()
_CN.LOFTR_LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.LOFTR_LOSS.COARSE_WEIGHT = 1.0
_CN.LOFTR_LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2', 'smooth_l1_with_std']
_CN.LOFTR_LOSS.FINE_WEIGHT = 1.0
# focal loss (coarse)
_CN.LOFTR_LOSS.SPG_SPVS = False
_CN.LOFTR_LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTR_LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTR_LOSS.POS_WEIGHT = 1.0
_CN.LOFTR_LOSS.NEG_WEIGHT = 1.0
_CN.LOFTR_LOSS.DUAL_SOFTMAX = False  # whether coarse-level use dual-softmax or not.
# coarse prior loss (optional)
_CN.LOFTR_LOSS.COARSE_PRIOR = CN()
_CN.LOFTR_LOSS.COARSE_PRIOR.ENABLE = False
_CN.LOFTR_LOSS.COARSE_PRIOR.FOCAL_ALPHA = 0.25
_CN.LOFTR_LOSS.COARSE_PRIOR.FOCAL_GAMMA = 2.0
# smooth_l1_with_std (fine)
_CN.LOFTR_LOSS.FINE_SMOOTH_L1_BETA = 1.0
_CN.LOFTR_LOSS.FINE_LOSS_WEIGHT = 1.0
_CN.LOFTR_LOSS.FINE_CORRECT_THR = 1.0
# fine-rejection loss (optional)
_CN.LOFTR_LOSS.FINE_REJECTION = CN()
_CN.LOFTR_LOSS.FINE_REJECTION.ENABLE = False
_CN.LOFTR_LOSS.FINE_REJECTION.FOCAL_ALPHA = 0.25
_CN.LOFTR_LOSS.FINE_REJECTION.FOCAL_GAMMA = 2.0

##############  ↓  LoFTR SfM  ↓  ################
_CN.LOFTR_SFM = CN()
# TODO: remove
_CN.LOFTR_SFM.REFINEMENT_METHOD='both' # choice:['direct_method', 'feature_based_method', 'both', 'neither'] NOTE: debug parameter TODO: remove in future

_CN.LOFTR_SFM.POSE_INITIAL = CN()
_CN.LOFTR_SFM.POSE_INITIAL.POSE_ESTIMATION_METHOD = "DEGENSAC"
_CN.LOFTR_SFM.POSE_INITIAL.RANSAC_PIXEL_THR = 1.0
_CN.LOFTR_SFM.POSE_INITIAL.RANSAC_CONF = 0.99999
_CN.LOFTR_SFM.POSE_INITIAL.RANSAC_MAX_ITERS = 100000
_CN.LOFTR_SFM.REFINEMENT = CN()
_CN.LOFTR_SFM.REFINEMENT.REFINEMENT_ITERATION_NUMBER = 3
_CN.LOFTR_SFM.REFINEMENT.NORM_FEATURE = False
_CN.LOFTR_SFM.REFINEMENT.VERBOSE = True
_CN.LOFTR_SFM.REFINEMENT.RESIDUAL_FORMAT = 'L*1' # choice: ['L*1', 'L*C']

# TODO: remove this part to function head!
# coarse level feature refinement configs
_CN.LOFTR_SFM.REFINEMENT.USE_COARSE_FEATURE_REFINEMENT = True
_CN.LOFTR_SFM.REFINEMENT.COARSE_LEVEL_SOLVER_TYPE = 'second_order' # choice: ['first_order', 'second_order']
_CN.LOFTR_SFM.REFINEMENT.COARSE_LEVEL_REFINE_TYPE = 'pose_depth' # choice: ['pose_depth', 'depth_pose', 'only_depth', 'only_pose']

# fine level feature refinement configs
_CN.LOFTR_SFM.REFINEMENT.USE_FINE_FEATURE_REFINEMENT = True
_CN.LOFTR_SFM.REFINEMENT.RECENTER_FINE_FEATURE = False
_CN.LOFTR_SFM.REFINEMENT.FINE_PATCH_FEATURE_SIZE = 5
_CN.LOFTR_SFM.REFINEMENT.FINE_LEVEL_SOLVER_TYPE = 'first_order' # choice: ['first_order', 'second_order', 'both']
_CN.LOFTR_SFM.REFINEMENT.FINE_LEVEL_REFINE_TYPE = 'depth_pose' # choice: ['pose_depth', 'depth_pose', 'only_depth', 'only_pose']
_CN.LOFTR_SFM.REFINEMENT.FINE_LEVEL_LM_RADIUS = 1e-2 # fine level refinement LM radius should be small


_CN.LOFTR_SFM.REFINEMENT.LM = CN()
_CN.LOFTR_SFM.REFINEMENT.LM.MIN_DIAGONAL = 1e-6
_CN.LOFTR_SFM.REFINEMENT.LM.MAX_DIAGONAL = 1e32
_CN.LOFTR_SFM.REFINEMENT.LM.ETA = 1e-3
_CN.LOFTR_SFM.REFINEMENT.LM.RESIDUAL_RESET_PERIOD = 10
_CN.LOFTR_SFM.REFINEMENT.LM.MAX_LINEAR_ITERATIONS = 150
_CN.LOFTR_SFM.REFINEMENT.LM.MAX_NUM_ITERATIONS = 100
_CN.LOFTR_SFM.REFINEMENT.LM.MAX_SUCCESS_ITERATIONS = 50
_CN.LOFTR_SFM.REFINEMENT.LM.MAX_INVALID_STEP = 5
_CN.LOFTR_SFM.REFINEMENT.LM.MIN_RELATIVE_DECREASE = 1e-3
_CN.LOFTR_SFM.REFINEMENT.LM.PARAMETER_TOLERANCE = 1e-8
_CN.LOFTR_SFM.REFINEMENT.LM.FUNCTION_TOLERANCE = 1e-6
_CN.LOFTR_SFM.REFINEMENT.LM.RAIDUS = 1e4
_CN.LOFTR_SFM.REFINEMENT.LM.SPAN = 4000000
_CN.LOFTR_SFM.REFINEMENT.LM.STEP_SCALE = 1.0

_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER = CN()
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.OPTIMIZER = 'Adam' # choice['Adam', 'SGD', 'RMSprop']
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.POSE = CN()
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.POSE.LR = 1e-3  # 8e-4 pose refinement learning rate need to be relative small
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.POSE.MAX_STEPS = 30
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.POSE.DECAY = 0.6 # decay rate across each optimization process
# Following paras are for other optimizer
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.POSE.MOMENTUM = 0.9
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.POSE.WEIGHT_DECAY = 0.9

_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.DEPTH = CN()
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.DEPTH.LR = 2e-2 # 2e-2
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.DEPTH.MAX_STEPS = 30
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.DEPTH.DECAY = 0.8
# Following paras are for other optimizer
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.DEPTH.MOMENTUM = 0.9
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.DEPTH.WEIGHT_DECAY = 0.9

_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.BA = CN()
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.BA.POSE_LR = 1e-3
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.BA.DEPTH_LR = 2e-2
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.BA.MAX_STEPS = 70
# Following paras are for other optimizer
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.BA.MOMENTUM = 0.9
_CN.LOFTR_SFM.REFINEMENT.FIRST_ORDER.BA.WEIGHT_DECAY = 0.9


##############  ↓  LoCTR Pipeline  ↓  ##############
# SuperPoint config is identical in SPG Pipeline
# 1. LoCTR
_CN.LOCTR = CN()
_CN.LOCTR.EXTRACTOR = 'SuperPoint'
_CN.LOCTR.EXTRACTOR_DS_TYPE = '2x2' # ['2x2', '3x3'], i.e. max-pooling or strided-conv
_CN.LOCTR.DESCRIPTOR_DIM = 256
# _CN.LOCTR.KEYPOINT_ENCODER = [32, 64, 128, 256]
_CN.LOCTR.ATTENTION = 'linear'
_CN.LOCTR.ATTENTION_LAYER = ['self', 'cross'] * 6
_CN.LOCTR.CMATCH_TYPE = 'sinkhorn'  # ['pw_cls', 'sinkhorn']
_CN.LOCTR.INIT_BIN_SCORE = 0.1
_CN.LOCTR.SINKHORN_ITERS = 5
_CN.LOCTR.MATCH_THRESHOLD = 0.2

# 2. LoCTR Loss
_CN.LOCTR.LOSS = CN()
_CN.LOCTR.LOSS.WEIGHT_COARSE = 1.0

# LOCTR: DEBUG
_CN.LOCTR.VIS_GT = False

##############  ↑  End of Pipeline ↑  ##############


##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = "ScanNet"  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = "ScanNet"
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
_CN.DATASET.USE_PETREL = True  # use Petrel-OSS data storage
_CN.DATASET.MIN_OVERLAP_SCORE = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'FDA']
_CN.DATASET.AUG_REF_ROOT = 'assets/isrf'
# ScanNet
_CN.DATASET.IMG_RESIZE = [640, 480]
# MegaDepth
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_DEPTH_PAD_SIZE = 2000  # pad the depth map to square. 2000 is the max size in MegaDepth.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DIVISIBLE_FACTOR = 8
_CN.DATASET.LOAD_KEYPOINTS = False

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adam"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.05

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning-rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # 'confidence' / 'evaluation'
_CN.TRAINER.PLOT_NMATCHES = 400

_CN.TRAINER.VIS_ALL_PAIRS = False

_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 1.0
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for data loader
# the `scene_balance` sampler will sample data with data samples
# from each scenes balanced during each epoch.
# ['scene_balance', 'random', 'normal', 'none'] - 'normal' for debuging / overfitting
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SHUFFLE_WITHIN_EPOCH_SUBSET = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.REPEAT_SAMPLE = 1  # repeat the sampled data this times for training. ()
# 'random' config
_CN.TRAINER.REPLACEMENT = False
_CN.TRAINER.NUM_SAMPLES = None

_CN.TRAINER.USE_FAST_DATALOADER = True

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed value might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
