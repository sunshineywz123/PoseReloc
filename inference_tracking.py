import time
import cv2
import glob
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


def get_default_paths(cfg):
    data_dir = cfg.input.data_dir
    sfm_model_dir = cfg.input.sfm_model_dir
    anno_dir = osp.join(sfm_model_dir, f'outputs_{cfg.match_type}_{cfg.network.detection}_{cfg.network.matching}/anno')
    avg_anno_3d_path = osp.join(anno_dir, 'anno_3d_average.npz')
    clt_anno_3d_path = osp.join(anno_dir, 'anno_3d_collect.npz')
    idxs_path = osp.join(anno_dir, 'idxs.npy')

    img_lists = []
    color_dir = osp.join(data_dir, 'color')
    img_lists += glob.glob(color_dir + '/*.png', recursive=True)

    intrin_full_path = osp.join(data_dir, 'intrinsics.txt')
    paths = {
        'data_dir': data_dir,
        'sfm_model_dir': sfm_model_dir,
        'avg_anno_3d_path': avg_anno_3d_path,
        'clt_anno_3d_path': clt_anno_3d_path,
        'idxs_path': idxs_path,
        'intrin_full_path': intrin_full_path
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
        'keypoints2d': keypoints2d[None].cuda(),  # [1, n1, 2]
        'keypoints3d': keypoints3d[None].cuda(),  # [1, n2, 3]
        'descriptors2d_query': descriptors2d[None].cuda(),  # [1, dim, n1]
        'descriptors3d_db': avg_descriptors3d[None].cuda(),  # [1, dim, n2]
        'descriptors2d_db': clt_descriptors[None].cuda(),  # [1, dim, n2*num_leaf]
        'image_size': image_size
    }

    return inp_data


def vis_reproj(paths, img_path, pose_pred, pose_gt, save_img=False, demo_dir=None):
    """ Draw 2d box reprojected by 3d box"""
    from src.utils.objScanner_utils import parse_3d_box, parse_K
    from src.utils.vis_utils import reproj, draw_3d_box

    box_3d_path = get_3d_box_path(paths['data_dir'])
    box_3d, box3d_homo = parse_3d_box(box_3d_path)

    intrin_full_path = paths['intrin_full_path']
    K_full, K_full_homo = parse_K(intrin_full_path)

    image_full_path = get_img_full_path(img_path)
    image_full = cv2.imread(image_full_path)

    # Draw gt 3d box
    reproj_box_2d_gt = reproj(K_full, pose_gt, box_3d)
    draw_3d_box(image_full, reproj_box_2d_gt, color='y')

    # Draw pred 3d box
    if pose_pred is not None:
        reproj_box_2d_pred = reproj(K_full, pose_pred, box_3d)
        draw_3d_box(image_full, reproj_box_2d_pred, color='g')

    if save_img:
        assert demo_dir, "Please assign a directory for saving results."
        img_idx = int(osp.basename(img_path).split('.')[0])
        obj_name = img_path.split('/')[-3]
        save_dir = osp.join(demo_dir, obj_name)
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        save_path = osp.join(save_dir, '{:05d}.jpg'.format(img_idx))
        cv2.imwrite(save_path, image_full)

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
    if cfg.use_tracker:
        from src.tracker import BATracker
        tracker = BATracker(cfg)
        track_interval = 3
    else:
        tracker = None
        track_interval = -1

    trained_model, extractor_model = load_model(cfg)
    img_lists, paths = get_default_paths(cfg)
    im_ids = [int(osp.basename(i).replace('.png', '')) for i in img_lists]
    im_ids.sort()
    img_lists = [osp.join(osp.dirname(img_lists[0]), f'{im_id}.png') for im_id in im_ids]

    dataset = HLOCDataset(img_lists, confs[cfg.network.detection]['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)
    evaluator = Evaluator()

    # anno_3d = get_3d_anno(paths['anno_3d_path'])
    idx = 0
    num_leaf = cfg.num_leaf
    time_cost = 0

    # with open(paths['avg_anno_3d_path'], 'r') as f:
    # avg_data = json.load(f)
    # with open(paths['clt_anno_3d_path'], 'r') as f:
    # clt_data = json.load(f)
    avg_data = np.load(paths['avg_anno_3d_path'])
    clt_data = np.load(paths['clt_anno_3d_path'])

    for data in tqdm.tqdm(loader):
        img_path = data['path'][0]
        inp = data['image'].cuda()

        # feature extraction
        torch.cuda.synchronize()
        start = time.time()
        pred_detection = extractor_model(inp)
        pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

        # posereloc inference
        inp_data = pack_data(pred_detection, avg_data, clt_data,
                             paths['idxs_path'], num_leaf, data['size'])
        pred, _ = trained_model(inp_data)

        matches = pred['matches0'].detach().cpu().numpy()
        valid = matches > -1

        kpts2d_q = pred_detection['keypoints']
        kpts3d_db = inp_data['keypoints3d'][0].detach().cpu().numpy()
        # kpts3d = anno_3d['keypoints3d'][0].detach().cpu().numpy()
        confidence = pred['matching_scores0'].detach().cpu().numpy()
        mkpts2d_q, mkpts3d_db, mconf = kpts2d_q[valid], kpts3d_db[matches[valid]], confidence[valid]

        # valid_detection = dict()
        # valid_detection['descriptors'] = pred_detection['descriptors'][:, valid]
        # valid_detection['keypoints'] = pred_detection['keypoints'][valid]
        # valid_detection['scores'] = pred_detection['scores'][valid]
        # pred_detection = valid_detection

        # evaluate
        intrin_path = get_intrin_path(img_path)
        K_crop = np.loadtxt(intrin_path)
        pose_pred, pose_pred_homo, inliers = ransac_PnP(K_crop, mkpts2d_q, mkpts3d_db, scale=1000)

        gt_pose_path = get_gt_pose_path(img_path)
        pose_gt = np.loadtxt(gt_pose_path)

        torch.cuda.synchronize()
        end = time.time()
        time_cost += end - start
        evaluator.evaluate(pose_pred, pose_gt)

        # Gather keyframe information and tracking
        if cfg.use_tracker:
            # TODO: augment 3d point_ids with invalid points been -1
            mkpts3d_db_inlier = mkpts3d_db[inliers.flatten()]
            mkpts2d_q_inlier = mkpts2d_q[inliers.flatten()]

            n_kpt = kpts2d_q.shape[0]

            valid_query_id = np.where(valid != False)[0][inliers.flatten()]
            kpts3d_full = np.ones([n_kpt, 3]) * 10086
            kpts3d_full[valid_query_id] = mkpts3d_db_inlier
            kpt3d_ids = matches[valid][inliers.flatten()]

            kf_dict = {
                'im_path': img_path,
                'kpt_pred': pred_detection,
                'valid_mask': valid,
                'mkpts2d': mkpts2d_q_inlier,
                'mkpts3d': mkpts3d_db_inlier,
                'kpt3d_full': kpts3d_full,
                'inliers': inliers,
                'kpt3d_ids': kpt3d_ids,
                'valid_query_id': valid_query_id,
                'pose_pred': pose_pred,
                'pose_gt': pose_gt,
                'K': K_crop
            }

            frame_dict = {
                'im_path': img_path,
                'kpt_pred': pred_detection,
                'pose_gt': pose_gt,
                'K': K_crop,
                'data': data
            }
            if idx % track_interval == 0:
                tracker.add_kf(kf_dict)
            else:
                tracker.track(frame_dict)

        # visualize
        image_full = vis_reproj(paths, img_path, pose_pred_homo, pose_gt, save_img=cfg.save_demo, demo_dir=cfg.demo_dir)

        mkpts3d_2d = reproj(K_crop, pose_gt, mkpts3d_db)
        image0 = Image.open(img_path).convert('LA')
        image1 = image0.copy()
        dump_vis3d(idx, cfg, image0, image1, image_full,
                   mkpts2d_q, mkpts3d_2d, mconf, inliers)

        idx += 1

    evaluator.summarize()
    time_cost += end - start
    print('=> average time cost: ', time_cost / len(dataset))


@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
