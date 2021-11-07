import time
import cv2
import glob
import tqdm
from matplotlib.pyplot import axis
import torch
import hydra
import tqdm
import json
import os.path as osp
import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig
from src.utils import data_utils


def get_img_full_path(img_path):
    return img_path.replace('/color/', '/color_full/')

def get_gt_pose_path(img_path):
    return img_path.replace('/color/', '/poses_ba/').replace('.png', '.txt')

def get_intrin_path(img_path):
    return img_path.replace('/color/', '/intrin_ba/').replace('.png', '.txt')

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


def get_default_paths(cfg):
    data_dir = cfg.input.data_dir
    sfm_model_dir = cfg.input.sfm_model_dir
    anno_dir = osp.join(sfm_model_dir, f'outputs_{cfg.match_type}_{cfg.network.detection}_{cfg.network.matching}/anno')
    avg_anno_3d_path = osp.join(anno_dir, 'anno_3d_average.npz')
    clt_anno_3d_path = osp.join(anno_dir, 'anno_3d_collect.npz')
    idxs_path = osp.join(anno_dir, 'idxs.npy')

    seq_id = data_dir.split('/')[-1].split('-')[-1]
    seq1_path = data_dir.replace(seq_id, '1')
    box_path = osp.join(seq1_path, 'Box.txt')
    trans_box_path = osp.join(seq1_path, 'Box_trans.txt')
    
    img_lists = []
    color_full_dir = osp.join(data_dir, 'color_full')
    img_lists += glob.glob(color_full_dir + '/*.png', recursive=True)

    intrin_full_path = osp.join(data_dir, 'intrinsics.txt')
    paths = {
        'data_dir': data_dir,
        'sfm_model_dir': sfm_model_dir,
        'avg_anno_3d_path': avg_anno_3d_path,
        'clt_anno_3d_path': clt_anno_3d_path,
        'idxs_path': idxs_path,
        'intrin_full_path': intrin_full_path,
        'box_path': box_path,
        'trans_box_path': trans_box_path
    }
    return img_lists, paths


def load_model(cfg):
    """ Load pretrained model """

    def load_trained_model(model_path):
        """ Load posereloc model """
        # from src.models.spg_model import LitModelSPG
        from src.models.GATs_spg_model import LitModelGATsSPG

        trained_model = LitModelGATsSPG.load_from_checkpoint(checkpoint_path=model_path)
        trained_model.cuda()
        trained_model.eval()
        trained_model.freeze()

        return trained_model
    
    def load_extractor_model(cfg, model_path):
        """ Load extractor model(SuperGlue) """
        from src.models.extractors.SuperPoint.superpoint_v1 import SuperPoint
        from src.hloc.extract_features import confs
        from src.utils.model_io import load_network
    
        extractor_model = SuperPoint(confs[cfg.network.detection]['conf'])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path)

        return extractor_model
    
    trained_model = load_trained_model(cfg.model.pretrain_model_path)
    extractor_model = load_extractor_model(cfg, cfg.model.extractor_model_path)
    return trained_model, extractor_model


def pack_data(detection, avg_data, clt_data, idxs_file, num_leaf, image_size):
    """ prepare data for posereloc inference """
    # with open(avg_anno_3d_file, 'r') as f:
    #     avg_data = json.load(f)
    
    # with open(clt_anno_3d_file, 'r') as f:
    #     clt_data = json.load(f)
    
    idxs = np.load(idxs_file)
    
    keypoints3d = torch.Tensor(clt_data['keypoints3d']).cuda()
    avg_descriptors3d = torch.Tensor(avg_data['descriptors3d'])
    clt_descriptors = torch.Tensor(clt_data['descriptors3d'])
    avg_scores3d = torch.Tensor(avg_data['scores3d'])
    clt_scores = torch.Tensor(clt_data['scores3d'])

    num_3d = keypoints3d.shape[0]
    avg_descriptors3d, avg_scores3d = data_utils.pad_features3d_random(avg_descriptors3d, avg_scores3d, num_3d)
    clt_descriptors, clt_scores = data_utils.build_features3d_leaves(clt_descriptors, clt_scores, idxs,
                                                                     num_3d, num_leaf)

    keypoints2d = torch.Tensor(detection['keypoints'])
    descriptors2d = torch.Tensor(detection['descriptors'])
    scores2d = torch.Tensor(detection['scores'])
    
    inp_data = {
        'keypoints2d': keypoints2d[None].cuda(), # [1, n1, 2] 
        'keypoints3d': keypoints3d[None].cuda(), # [1, n2, 3]
        'descriptors2d_query': descriptors2d[None].cuda(), # [1, dim, n1]
        'descriptors3d_db': avg_descriptors3d[None].cuda(), # [1, dim, n2]
        'descriptors2d_db': clt_descriptors[None].cuda(), # [1, dim, n2*num_leaf]
        'image_size': image_size
    }

    return inp_data


def read_trans(trans_box_path):
    f = open(trans_box_path, 'r')
    line = f.readlines()[1]

    data = [float(e) for e in line.split(' ')]
    scale = np.array(data[0])
    rot_vec = np.array(data[1:4])
    trans_vec = np.array(data[4:])

    return scale, rot_vec, trans_vec


def trans_box(orig_box, scale, rot_vec, trans_vec):
    scaled_box = orig_box * scale 

    trans = np.eye(4)
    trans[:3, :3] = cv2.Rodrigues(rot_vec)[0]
    trans[:3, 3:] = trans_vec.reshape(3, 1)

    scaled_box_homo = np.concatenate([scaled_box, np.ones((scaled_box.shape[0], 1))], axis=-1).T
    trans_box_homo = trans @ scaled_box_homo

    trans_box_homo[:3, :] /= trans_box_homo[3:, :]
    return trans_box_homo[:3].T
        

def vis_reproj(paths, img_path, pose_pred, save_img=False, demo_dir=None):
    """ Draw 2d box reprojected by 3d box"""
    from src.utils.objScanner_utils import parse_3d_box, parse_K
    from src.utils.vis_utils import reproj, draw_3d_box

    box_path = paths['box_path']
    trans_box_path = paths['trans_box_path']
    
    box_3d, box3d_homo = parse_3d_box(box_path)
    scale, rot_vec, trans_vec = read_trans(trans_box_path)
    box_3d_trans = trans_box(box_3d, scale, rot_vec, trans_vec)
    
    intrin_full_path = paths['intrin_full_path']
    K_full, K_full_homo = parse_K(intrin_full_path)

    # image_full_path = get_img_full_path(img_path)
    image_full = cv2.imread(img_path)

    # Draw pred 3d box
    if pose_pred is not None:
        reproj_box_2d_pred = reproj(K_full, pose_pred, box_3d_trans)
        draw_3d_box(image_full, reproj_box_2d_pred, color='g')

    if save_img:
        assert demo_dir, "Please assign a directory for saving results."
        img_idx = int(osp.basename(img_path).split('.')[0])
        obj_name = img_path.split('/')[-3]
        save_dir = osp.join(demo_dir, obj_name) 
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        save_path = osp.join(save_dir, '{:05d}.jpg'.format(img_idx))
        cv2.imwrite(save_path, image_full)
        # import ipdb; ipdb.set_trace()

    return image_full


def dump_vis3d(idx, cfg, img_full):
    """ Visualize by vis3d """
    from vis3d import Vis3D
    
    seq_name = '_'.join(cfg.input.data_dir.split('/')[-2:])
    if cfg.suffix:
        seq_name += '_' + cfg.suffix
    vis3d = Vis3D(cfg.output.vis_dir, seq_name)
    vis3d.set_scene_id(idx)

    image_full_pil = Image.fromarray(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))
    vis3d.add_image(image_full_pil, name='results')


def prepare_data(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img.astype(np.float32)
    img_size = img.shape[:2]
    img = img[None]
    img /= 255.

    inp = torch.Tensor(img)[None].cuda()
    return inp, np.array(img_size)[None]
    

@torch.no_grad()
def inference(cfg):
    """ Inference & visualize"""
    from src.datasets.hloc_dataset import HLOCDataset
    from src.hloc.extract_features import confs
    from src.utils.vis_utils import reproj, ransac_PnP
    from src.evaluators.spg_evaluator import Evaluator
    from src.utils.yolo_utils import process_img, non_max_suppression, scale_coords
    from src.utils.arkit_utils import get_K

    obj_name = cfg.input.sfm_model_dir.split('/')[-1]
    yolo_model_path = cfg.model.detection_model_path.format(obj_name)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=yolo_model_path)
    stride = int(yolo_model.stride.max())
    yolo_det_size = cfg.yolo_det_size

    trained_model, extractor_model = load_model(cfg)
    img_lists, paths = get_default_paths(cfg)
    
    idx = 0
    num_leaf = cfg.num_leaf
    
    avg_data = np.load(paths['avg_anno_3d_path'])
    clt_data = np.load(paths['clt_anno_3d_path'])

    K, K_homo = get_K(paths['intrin_full_path'])
    for img_file in tqdm.tqdm(img_lists):
        # prepare yolo forward data
        img_full = cv2.imread(img_file)
        inp_yolo = process_img(img_full, yolo_det_size, stride)

        pred_yolo = yolo_model(inp_yolo)[0]

        pred_yolo = non_max_suppression(pred_yolo)

        if pred_yolo[0].shape[0] == 0: # No obj is detected
            continue
        
        for i, det in enumerate(pred_yolo):
            det[:, :4] = scale_coords(inp_yolo.shape[2:], det[:, :4], img_full.shape).round()
            box = det[0, :4].cpu().numpy().astype(np.int)

        x0, y0, x1, y1 = box
        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop, K_crop_homo = data_utils.get_K_crop_resize(box, K, resize_shape)
        image_crop, trans1 = data_utils.get_image_crop_resize(img_full, box, resize_shape)

        box_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([512, 512])
        K_crop, K_crop_homo = data_utils.get_K_crop_resize(box_new, K_crop, resize_shape)
        image_crop, trans2 = data_utils.get_image_crop_resize(image_crop, box_new, resize_shape)

        inp_spp, img_size = prepare_data(image_crop)
        pred_detection = extractor_model(inp_spp)

        pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

        inp_data = pack_data(pred_detection, avg_data, clt_data, paths['idxs_path'], num_leaf, img_size)
        pred, _ = trained_model(inp_data)

        matches = pred['matches0'].detach().cpu().numpy()
        valid = matches > -1
        kpts2d = pred_detection['keypoints']
        kpts3d = inp_data['keypoints3d'][0].detach().cpu().numpy()
        confidence = pred['matching_scores0'].detach().cpu().numpy()
        mkpts2d, mkpts3d, mconf = kpts2d[valid], kpts3d[matches[valid]], confidence[valid]

        pose_pred, pose_pred_homo, inliers = ransac_PnP(K_crop, mkpts2d, mkpts3d, scale=1000)
        # visualize
        img_full = vis_reproj(paths, img_file, pose_pred, save_img=cfg.save_demo, demo_dir=cfg.demo_dir)
        idx += 1
        
        if cfg.save_vis3d:
            dump_vis3d(idx, cfg, img_full)



@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)

            # reproj_box3d_homo = np.concatenate([reproj_box3d, np.ones((reproj_box3d.shape[0], 1))], axis=-1).T
if __name__ == "__main__":
    main()