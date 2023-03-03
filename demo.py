import hydra
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from loguru import logger
import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module, this line should before import torch
import os.path as osp
import glob
import numpy as np
import natsort
from pathlib import Path
import torch

from src.utils import data_utils
from src.utils import vis_utils
from src.utils.metric_utils import ransac_PnP
from src.datasets.OnePosePlus_inference_dataset import OnePosePlusInferenceDataset
from src.inference.inference_OnePosePlus import build_model
from src.local_feature_object_detector.local_feature_2D_detector import LocalFeatureObjectDetector
import json
import glob
import cv2
from PIL import Image

import ipdb 
import sys

def get_img_full_path(img_path):
    return img_path.replace('/color/', '/color_full/')

def get_gt_pose_path(img_path):
    return img_path.replace('/color/', '/poses/').replace('.png', '.txt')

def get_intrin_path(img_path):
    return img_path.replace('/color/', '/intrin/').replace('.png', '.txt')

def get_3d_box_path(data_dir):
    refined_box_path = osp.join(data_dir, 'RefinedBox.txt')
    box_path = refined_box_path if osp.isfile(refined_box_path) else osp.join(data_dir, 'Box.txt')
    return box_path

def get_3d_anno(anno_3d_path):
    """ Read 3d information about this seq """
    with open(anno_3d_path, 'r') as f:
        anno_3d = json.load(f)
    
    descriptors3d = torch.Tensor(anno_3d['descriptors3d'])[None].cuda()
    keypoints3d = torch.Tensor(anno_3d['keypoints3d'])[None].cuda()
    scores3d = torch.Tensor(anno_3d['scores3d'])[None].cuda()
    anno_3d = {
        'keypoints3d': keypoints3d,
        'descriptors3d': descriptors3d,
        'scores3d': scores3d
    }
    return anno_3d

def vis_reproj(paths, img_path, pose_pred, pose_gt):
    """ Draw 2d box reprojected by 3d box"""
    from src.utils.objScanner_utils import parse_3d_box, parse_K
    from src.utils.vis_utils import reproj, draw_3d_box

    box_3d = np.loadtxt(paths["bbox3d_path"])

    intrin_full_path = paths['intrin_full_path']
    K_full, K_full_homo = parse_K(intrin_full_path)

    image_full_path = get_img_full_path(img_path)
    image_full = cv2.imread(image_full_path)

    reproj_box_2d_gt = reproj(K_full, pose_gt, box_3d)
    draw_3d_box(image_full, reproj_box_2d_gt, color='y')
    if pose_pred is not None:
        reproj_box_2d_pred = reproj(K_full, pose_pred, box_3d)
        draw_3d_box(image_full, reproj_box_2d_pred, color='g')

    return image_full


def dump_vis3d(idx, paths, image0, image1, image_full,
               kpts2d, kpts2d_reproj, confidence, inliers):
    """ Visualize by vis3d """
    from vis3d import Vis3D
    
    vis3d = Vis3D(paths["vis_dir"], 'test')
    vis3d.set_scene_id(idx)

    # property for vis3d
    reproj_distance = np.linalg.norm(kpts2d_reproj - kpts2d, axis=1)
    inliers_bool = np.zeros((kpts2d.shape[0], 1), dtype=np.bool)
    if inliers is not None:
        inliers_bool[inliers] = True
        num_inliers = len(inliers)
    else:
        num_inliers = 0
    
    vis3d.add_keypoint_correspondences(image0, image1, kpts2d, kpts2d_reproj,
                                       metrics={
                                           'mconf': confidence.tolist(),
                                           'reproj_distance': reproj_distance.tolist()
                                       },
                                       booleans={
                                           'inliers': inliers_bool.tolist()
                                           
                                       },
                                       meta={
                                           'num_inliers': num_inliers,
                                           'width': image0.size[0],
                                           'height': image0.size[1],
                                       },
                                       name='matches')  
    image_full_pil = Image.fromarray(cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB))
    vis3d.add_image(image_full_pil, name='results')

def get_default_paths(cfg, data_root, data_dir, sfm_model_dir):
    sfm_ws_dir = osp.join(
        sfm_model_dir,
        "sfm_ws",
        "model",
    )

    img_lists = []
    color_dir = osp.join(data_dir, "color_full")
    img_lists += glob.glob(color_dir + "/*.png", recursive=True)
    
    img_lists = natsort.natsorted(img_lists)

    # Visualize detector:
    vis_detector_dir = osp.join(data_dir, "detector_vis")
    if osp.exists(vis_detector_dir):
        os.system(f"rm -rf {vis_detector_dir}")
    os.makedirs(vis_detector_dir, exist_ok=True)
    det_box_vis_video_path = osp.join(data_dir, "det_box.mp4")

    # Visualize pose:
    vis_box_dir = osp.join(data_dir, "pred_vis")
    if osp.exists(vis_box_dir):
        os.system(f"rm -rf {vis_box_dir}")
    os.makedirs(vis_box_dir, exist_ok=True)
    demo_video_path = osp.join(data_dir, "demo_video.mp4")

    # intrin_full_dir = osp.join(data_dir, "origin_intrin")
    intrin_full_path = osp.join(data_dir, "intrinsics.txt")
    intrin_full_dir = osp.join(data_dir, 'intrin_full')

    bbox3d_path = osp.join(data_root, 'box3d_corners.txt')
    
    # Visualize vis:
    vis_dir = osp.join(data_dir, "vis_dir")
    if osp.exists(vis_dir):
        os.system(f"rm -rf {vis_dir}")
    os.makedirs(vis_dir, exist_ok=True)
    # Visualize vis debug:
    vis_debug_dir = osp.join(data_dir, "debug")
    if osp.exists(vis_debug_dir):
        os.system(f"rm -rf {vis_debug_dir}")
    os.makedirs(vis_debug_dir, exist_ok=True)
    paths = {
        "data_root": data_root,
        "data_dir": data_dir,
        "sfm_dir": sfm_model_dir,
        "sfm_ws_dir": sfm_ws_dir,
        "bbox3d_path": bbox3d_path,
        "intrin_full_path": intrin_full_path,
        "intrin_full_dir": intrin_full_dir,
        "vis_detector_dir": vis_detector_dir,
        "vis_box_dir": vis_box_dir,
        "det_box_vis_video_path": det_box_vis_video_path,
        "demo_video_path": demo_video_path,
        "vis_dir": vis_dir,
        "debug": vis_debug_dir,
    }
    return img_lists, paths

def inference_core(cfg, data_root, seq_dir, sfm_model_dir):
    from src.utils.vis_utils import reproj
    img_list, paths = get_default_paths(cfg, data_root, seq_dir, sfm_model_dir)
    dataset = OnePosePlusInferenceDataset(
        paths['sfm_dir'],
        img_list,
        load_3d_coarse=cfg.datamodule.load_3d_coarse,
        shape3d=cfg.datamodule.shape3d_val,
        img_pad=cfg.datamodule.img_pad,
        img_resize=None,
        df=cfg.datamodule.df,
        pad=cfg.datamodule.pad3D,
        load_pose_gt=False,
        n_images=None,
        demo_mode=True
    )
    local_feature_obj_detector = LocalFeatureObjectDetector(
        sfm_ws_dir=paths["sfm_ws_dir"],
        output_results=False,
        detect_save_dir=paths["vis_detector_dir"],
    )
    match_2D_3D_model = build_model(cfg['model']["OnePosePlus"], cfg['model']['pretrained_ckpt'])
    match_2D_3D_model.cuda()

    K, _ = data_utils.get_K(paths["intrin_full_path"])

    bbox3d = np.loadtxt(paths["bbox3d_path"])
    pred_poses = {}  # {id:[pred_pose, inliers]}
    for id in tqdm(range(len(dataset))):
        data = dataset[id]
        query_image = data['query_image']
        query_image_path = data['query_image_path']

        debug = {'id':id,'paths':paths}
        # bbox, inp_crop, K_crop = local_feature_obj_detector.detect(query_image, query_image_path, K,debug=debug)
        # Detect object:
        if id == 0:
            # Detect object by 2D local feature matching for the first frame:
            bbox, inp_crop, K_crop = local_feature_obj_detector.detect(query_image, query_image_path, K,debug=debug)
        else:
            # Use 3D bbox and previous frame's pose to yield current frame 2D bbox:
            previous_frame_pose, inliers = pred_poses[id - 1]

            if len(inliers) < 20:
                # Consider previous pose estimation failed, reuse local feature object detector:
                bbox, inp_crop, K_crop = local_feature_obj_detector.detect(
                    query_image, query_image_path, K
                ,debug=debug)
            else:
                (
                    bbox,
                    inp_crop,
                    K_crop,
                ) = local_feature_obj_detector.previous_pose_detect(
                    query_image_path, K, previous_frame_pose, bbox3d
                )
        
        data.update({"query_image": inp_crop.cuda()})

        # Perform keypoint-free 2D-3D matching and then estimate object pose of query image by PnP:
        with torch.no_grad():
            match_2D_3D_model(data)
        mkpts_3d = data["mkpts_3d_db"].cpu().numpy() # N*3
        mkpts_query = data["mkpts_query_f"].cpu().numpy() # N*2
        mconf = data["mconf"].cpu().numpy() # N*2
        pose_pred, _, inliers, _ = ransac_PnP(K_crop, mkpts_query, mkpts_3d, scale=1000, pnp_reprojection_error=7, img_hw=[512,512], use_pycolmap_ransac=True)

        pred_poses[id] = [pose_pred, inliers]

        # Visualize:
        vis_utils.save_demo_image(
            pose_pred,
            K,
            image_path=query_image_path,
            box3d=bbox3d,
            draw_box=len(inliers) > 20,
            save_path=osp.join(paths["vis_box_dir"], f"{id}.jpg"),
        )
        
        # visualize
        image_full = vis_reproj(paths, query_image_path, pose_pred, pose_pred)

        mkpts3d_2d = reproj(K_crop, pose_pred, mkpts_3d)
        image0 = Image.open(query_image_path).convert('LA')
        image1 = image0.copy()
        dump_vis3d(id, paths, image0, image1, image_full,
                   mkpts_query, mkpts3d_2d, mconf, inliers) 
        
    # Output video to visualize estimated poses:
    logger.info(f"Generate demo video begin...")
    vis_utils.make_video(paths["vis_box_dir"], paths["demo_video_path"])

def inference(cfg):
    data_dirs = cfg.data_base_dir
    sfm_model_dirs = cfg.sfm_base_dir

    if isinstance(data_dirs, str) and isinstance(sfm_model_dirs, str):
        data_dirs = [data_dirs]
        sfm_model_dirs = [sfm_model_dirs]

    for data_dir, sfm_model_dir in tqdm(
        zip(data_dirs, sfm_model_dirs), total=len(data_dirs)
    ):
        splits = data_dir.split(" ")
        data_root = splits[0]
        for seq_name in splits[1:]:
            seq_dir = osp.join(data_root, seq_name)
            logger.info(f"Eval {seq_dir}")
            inference_core(cfg, data_root, seq_dir, sfm_model_dir)

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)
    # try:
    #    globals()[cfg.type](cfg)
    # except:
    #    type, value, traceback = sys.exc_info()
    #    ipdb.post_mortem(traceback)

if __name__ == "__main__":
        main()

