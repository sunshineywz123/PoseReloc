from src.NeuralSfM.loftr_config.default import _CN as cfg
cfg.PIPELINE = 'loftr_sfm'

# model
cfg.LOFTR_MATCH_COARSE.THR = 0.2  # train: 0.2; test: 0.4
cfg.LOFTR_MATCH_COARSE.SKH.PREFILTER = True
cfg.LOFTR_LOSS.FOCAL_ALPHA = 1.0
cfg.LOFTR_LOSS.COARSE_WEIGHT = 1.0
cfg.LOFTR_LOSS.FINE_WEIGHT = 0.25 * (cfg.LOFTR_FINE.WINDOW_SIZE/5) ** 2

# deep-skh3
cfg.LOFTR_BACKBONE.RESNETFPN.INITIAL_DIM = 128  # 64 is not supported in the version of BasicBlock
cfg.LOFTR_BACKBONE.RESNETFPN.BLOCK_DIMS = [128, 196, 256]
cfg.LOFTR_BACKBONE.RESNETFPN.BLOCK_TYPE = "BasicBlock"
cfg.LOFTR_COARSE.D_MODEL = 256
cfg.LOFTR_FINE.D_MODEL = 128

cfg.LOFTR_COARSE.LAYER_NAMES = ['self', 'cross'] * 4
cfg.LOFTR_FINE.LAYER_NAMES = ['self', 'cross'] * 1

# context
cfg.LOFTR_FINE.CONCAT_COARSE_FEAT = True

# use superpoint and grid as fine detector
cfg.LOFTR_MATCH_FINE.DETECTOR = 'SuperPoint and grid'
cfg.LOFTR_MATCH_FINE.SPP.KEYPOINT_THRESHOLD = 0.005
cfg.LOFTR_MATCH_FINE.SPP.MAX_KEYPOINTS = -1

cfg.LOFTR_FINE.WINDOW_SIZE = 11

# adamw
cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1  # 0.05: 0.6278; 0.01: 0.6157

# trainer
# TODO: Tune Learning Rate
cfg.TRAINER.CANONICAL_BS = 64
cfg.TRAINER.CANONICAL_LR = 4e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]
cfg.LOFTR_MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.23

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

# dual-softmax
cfg.LOFTR_MATCH_COARSE.TYPE = 'dual-softmax'

# Important config for this config!
cfg.LOFTR_MATCH_FINE.ENABLE = False
