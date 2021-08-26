from re import M
import cv2
import glob
import torch
import hydra
import tqdm
import json
import os.path as osp
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig


def get_img_full_path(img_path):
    return img_path.replace('/color/', '/color_full/')

def get_gt_pose_path(img_path):
    return img_path.replace('color', 'poses').replace('.png', '.txt')

def get_intrin_path(img_path):
    return img_path.replace('color', 'intrin').replace('.png', '.txt')

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


def get_default_paths(data_dir, sfm_model_dir, match_type):
    anno_dir = osp.join(sfm_model_dir, 'outputs_{}/anno'.format(match_type))
    anno_3d_path = osp.join(anno_dir, 'anno_3d.json')
    
    img_lists = []
    color_dir = osp.join(data_dir, 'color')
    img_lists += glob.glob(color_dir + '/*.png', recursive=True)

    intrin_full_path = osp.join(data_dir, 'intrinsics.txt')
    paths = {
        'data_dir': data_dir,
        'sfm_model_dir': sfm_model_dir,
        'anno_3d_path': anno_3d_path,
        'intrin_full_path': intrin_full_path
    }
    return img_lists, paths


def load_model(cfg):
    """ Load pretrained model """

    def load_trained_model(model_path):
        """ Load posereloc model """
        from src.models.spg_model import LitModelSPG

        trained_model = LitModelSPG.load_from_checkpoint(checkpoint_path=model_path)
        trained_model.cuda()
        trained_model.eval()
        trained_model.freeze()

        return trained_model
    
    def load_extractor_model(model_path):
        """ Load extractor model(SuperGlue) """
        from src.models.extractors.SuperPoint.superpoint_v1 import SuperPoint
        from src.hloc.extract_features import confs
        from src.utils.model_io import load_network
    
        extractor_model = SuperPoint(confs['spp_det']['conf'])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path)

        return extractor_model
    
    trained_model = load_trained_model(cfg.model.pretrain_model_path)
    extractor_model = load_extractor_model(cfg.model.extractor_model_path)
    return trained_model, extractor_model


def pack_data(detection, anno_3d, image_size):
    """ prepare data for posereloc inference """
    keypoints2d = torch.Tensor(detection['keypoints'])[None].float().cuda()
    descriptors2d = torch.Tensor(detection['descriptors'])[None].float().cuda()
    scores2d = torch.Tensor(detection['scores'])[None].float().cuda()
    
    inp_data = {
        'keypoints2d': keypoints2d,
        'keypoints3d': anno_3d['keypoints3d'],
        'descriptors2d': descriptors2d,
        'descriptors3d': anno_3d['descriptors3d'],
        'scores2d': scores2d.reshape(1, -1, 1),
        'scores3d': anno_3d['scores3d'],
        'image_size': image_size
    }
    return inp_data


def vis_reproj(paths, img_path, pose_pred, pose_gt):
    """ Draw 2d box reprojected by 3d box"""
    from src.utils.objScanner_utils import parse_3d_box, parse_K
    from src.utils.vis_utils import reproj, draw_3d_box

    box_3d_path = get_3d_box_path(paths['data_dir'])
    box_3d, box3d_homo = parse_3d_box(box_3d_path)

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


def dump_vis3d(idx, cfg, image0, image1, image_full,
               kpts2d, kpts2d_reproj, confidence, inliers):
    """ Visualize by vis3d """
    from vis3d import Vis3D
    
    seq_name = '_'.join(cfg.input.data_dir.split('/')[-2:])
    if cfg.suffix:
        seq_name += '_' + cfg.suffix
    vis3d = Vis3D(cfg.output.vis_dir, seq_name)
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


@torch.no_grad()
def inference(cfg):
    """ Inference & visualize"""
    from src.datasets.hloc_dataset import HLOCDataset
    from src.hloc.extract_features import confs
    from src.utils.vis_utils import reproj, ransac_PnP
    from src.evaluators.spg_evaluator import Evaluator
    
    trained_model, extractor_model = load_model(cfg)
    img_lists, paths = get_default_paths(cfg.input.data_dir, cfg.input.sfm_model_dir, cfg.match_type)

    dataset = HLOCDataset(img_lists, confs['spp_det']['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)
    evaluator = Evaluator() # todo

    anno_3d = get_3d_anno(paths['anno_3d_path'])
    idx = 0
    for data in tqdm.tqdm(loader):
        img_path = data['path'][0]
        inp = data['image'].cuda()
        
        # feature extraction
        pred_detection = extractor_model(inp)
        pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

        # posereloc inference
        inp_data = pack_data(pred_detection, anno_3d, data['size'])
        pred, _ = trained_model(inp_data)

        matches = pred['matches0'].detach().cpu().numpy()
        valid = matches > -1
        kpts2d = pred_detection['keypoints']
        kpts3d = anno_3d['keypoints3d'][0].detach().cpu().numpy()
        confidence = pred['matching_scores0'].detach().cpu().numpy()
        mkpts2d, mkpts3d, mconf = kpts2d[valid], kpts3d[matches[valid]], confidence[valid]

        # evaluate
        intrin_path = get_intrin_path(img_path)
        K_crop = np.loadtxt(intrin_path)
        pose_pred, pose_pred_homo, inliers = ransac_PnP(K_crop, mkpts2d, mkpts3d)
        
        gt_pose_path = get_gt_pose_path(img_path)
        pose_gt = np.loadtxt(gt_pose_path)
        evaluator.evaluate(pose_pred, pose_gt)

        # visualize
        image_full = vis_reproj(paths, img_path, pose_pred_homo, pose_gt)

        mkpts3d_2d = reproj(K_crop, pose_gt, mkpts3d)
        image0 = Image.open(img_path).convert('LA')
        image1 = image0.copy()
        dump_vis3d(idx, cfg, image0, image1, image_full,
                   mkpts2d, mkpts3d_2d, mconf, inliers) 
        
        idx += 1

    evaluator.summarize()
        

@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()