from .SuperPoint.superpoint import SuperPoint
from .SuperPoint.superpoint_every_cell import SuperPointEC
from .svcnn.svcnn import SVCNN
from .Disk.disk import DISK
from .D2_net.D2_net_test import D2Net
from .R2D2.R2D2 import R2D2
from .sift import SIFT


def build_extractor(config):
    detector_type = config['detector']
    if detector_type == 'OnGrid':
        return None
    elif detector_type in ['SuperPoint','SuperPoint and grid']:
        if config['feats_hdf5']:
            return None
        else:
            return SuperPoint(config['spp'])
    elif detector_type in ['SuperPointEC', 'SuperPointEC and grid']:
        """ Superpoint that predicts on every cell """
        return SuperPointEC(config['sppec'])
    elif detector_type in ['SIFT', 'SIFT and grid']:
        """ SIFT with a normal config / low-threshold config """
        _cfg = config['sift']
        cfg = {
            'nfeatures': _cfg['n_features'],
            'contrastThreshold': _cfg['contrast_threshold'],
            'edgeThreshold': _cfg['edge_threshold'],
        }
        return SIFT(cfg)
    else:
        raise NotImplementedError()

def build_extractor_for_spg(config):
    detector_type = config["detector_type"]
    if detector_type=="SuperPoint":
        return SuperPoint(config["detector"])
    elif detector_type=="SVCNN":
        return SVCNN(config["detector"])
    elif detector_type=="Disk":
        return DISK(config["detector"])
    elif detector_type=="D2_net":
        return D2Net(config["detector"])
    elif detector_type=="R2D2":
        return R2D2(config["detector"])
    else:
        raise NotImplementedError

def build_extractor_for_loctr(config):
    extractor_type = config['loctr']['extractor']
    if extractor_type == 'SuperPoint':
        return SuperPoint(config['superpoint'])
    else:
        raise NotImplementedError