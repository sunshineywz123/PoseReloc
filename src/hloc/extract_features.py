import cv2
import h5py
import tqdm
import torch
import logging

from torch.utils.data import DataLoader

confs = {
    'sift': {
        'output': 'feats-sift',
        'conf': {
            'keypoints_threshold': 0.02,
        }
    },
    'spp_det': {
        'output': 'feats-spp',
        'model': {
            'name': 'spp_det',
        },
        'preprocessing': {
            'grayscale': True,
            'resize_h': 512,
            'resize_w': 512
        },
        'conf': {
            'descriptor_dim': 256,
            'nms_radius': 3,
            'max_keypoints': 4096,
            'keypoints_threshold': 0.6
        }
    },
    'svcnn_det': {
        'output': 'feats-svcnn',
        'model': {
            'name': 'svcnn_det',
        },
        'preprocessing': {
            'grayscale': True,
            'resize_h': 512,
            'resize_w': 512,
        },
        'conf': {
            'keypoints_threshold': 0.02
        }
    }
}


@torch.no_grad()
def spp(img_lists, feature_out, cfg):
    """extract keypoints info by superpoint"""
    from src.utils.model_io import load_network
    from src.models.extractors.SuperPoint.superpoint_v1 import SuperPoint as spp_det
    from src.datasets.hloc_dataset import HLOCDataset
    
    conf = confs[cfg.network.detection]
    model = spp_det(conf['conf']).cuda()
    model.eval()
    load_network(model, cfg.network.detection_model_path)

    dataset = HLOCDataset(img_lists, conf['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)

    feature_file = h5py.File(feature_out, 'w')
    logging.info(f'Exporting features to {feature_out}')
    for data in tqdm.tqdm(loader):
        inp = data['image'].cuda()
        pred = model(inp)

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = data['size'][0].numpy()
        
        grp = feature_file.create_group(data['path'][0])
        for k, v in pred.items():
            grp.create_dataset(k, data=v)
        
        del pred
    
    feature_file.close()
    logging.info('Finishing exporting features.')


def sift(img_lists, feature_out, cfg):
    from src.models.extractors.sift.sift import SIFT
    conf = confs[cfg.network.detection]
    
    feature_file = h5py.File(feature_out, 'w')
    logging.info(f'Exporting features to {feature_out}')
    
    model = SIFT(conf['conf'])
    for img_path in tqdm.tqdm(img_lists):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pred = model(image)
        
        grp = feature_file.create_group(img_path)
        for k, v in pred.items():
            grp.create_dataset(k, data=v)
        
        del pred

    feature_file.close()
    logging.info('Finishing exporting features.')