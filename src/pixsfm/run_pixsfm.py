import ray
import os.path as osp
from pathlib import Path
from pixsfm.refine_hloc import PixSfM

def pixsfm(img_lists, deep_sfm_dir, empty_dir, covis_pairs_out, feature_out, matches_out, use_costmaps=False, patch_size=8, img_resize=1600):
    # NOTE: if 'use_cache" is opened, PixSfM will store extracted features, however will report error sequentially. open 'use_cache' only for store features and make plots
    conf = {"dense_features": {'patch_size': patch_size, 'max_edge': img_resize, "use_cache": False, "cache_format": 'chunked', "load_cache_on_init": True},"BA": {"optimizer": {
        "refine_focal_length": False,
        "refine_extra_params": False,  # distortion parameters
        "refine_extrinsics": False,    # camera poses
    }}}

    if use_costmaps:
        conf["BA"]['strategy'] = 'costmaps'

    refiner = PixSfM(conf)
    image_dir = osp.dirname(img_lists[0])
    refined, sfm_outputs = refiner.triangulation(output_dir=Path(osp.join(deep_sfm_dir, 'model')), reference_model_path=Path(empty_dir), image_dir=Path(image_dir), pairs_path=Path(covis_pairs_out), features_path=Path(feature_out), matches_path=Path(matches_out))
    # return refined, sfm_outputs

@ray.remote(num_cpus=2, num_gpus=1, max_calls=1)  # release gpu after finishing
def pixsfm_ray_wrapper(*args, **kwargs):
    return pixsfm(*args, **kwargs)