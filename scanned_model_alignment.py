
import argparse
import natsort
import os
import os.path as osp
import open3d as o3d
from tqdm import tqdm
import trimesh
import numpy as np
import json
from tools.data_prepare.render_cad_model_to_depth import render_cad_model_to_depth
from src.utils.metric_utils import ransac_PnP

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input:
    parser.add_argument(
        "--data_base_path", type=str, default="/nas/users/hexingyi/onepose_hard_data",
    )

    parser.add_argument(
        '--rendered_save_dir', type=str, default='/nas/users/hexingyi/onepose_model_align_rendered'
    )

    parser.add_argument(
        '--annotate_dir', type=str, default='/nas/users/hexingyi/onepose_model_align_annotation'
    )

    parser.add_argument(
        # "--obj_ids", nargs="+", default=['0601', '0604', '0606', '0607', '0623', '0627', '0639', '0640', '0641'],
        # "--obj_ids", nargs="+", default=["0600", '0601', '0604', '0606', '0607', '0623', '0627', '0639', '0640', '0641'], # OnePose hard

        # "--obj_ids", nargs="+", default=["0600", "0604", "0639"], # ok
        # "--obj_ids", nargs="+", default=["0600"], # ok
        # "--obj_ids", nargs="+", default=["0604"], # ok
        # "--obj_ids", nargs="+", default=["0606"], # need to re-annotate
        # "--obj_ids", nargs="+", default=["0607"], # need to re-annotate
        # "--obj_ids", nargs="+", default=["0639"], # ok NOTE: use model_aligned.ply
        "--obj_ids", nargs="+", default=["0601", "0623", "0627", "0640", "0641"], # need to re-annotate
        # "--obj_ids", nargs="+", default=["0801", "0802", "0804", "0805", "0806", "0808", "0809" ], # LINEMOD
    )

    args = parser.parse_args()
    return args

annotated_3d ={
    "0600": np.array(
        [
            [-0.009362,0.043476,-0.010770],
            [-0.002378,0.034912,0.005941],
            [0.004508,0.008389,0.012626],
            [0.006636,-0.013833,0.014215],
            [-0.022150,-0.007638,0.013156],
            [-0.020503,0.007741,0.011806],
        ]
    ),
    "0604": np.array(
        [
            [0.022855,0.057072,0.011356],
            [0.035346,-0.008798,0.034047],
            [0.032277,-0.071197,0.034481],
            [-0.038739,-0.070592,-0.013770],
            [-0.054629,0.018821,0.002816],
        ]
    ),
    "0606": np.array(
        [
            [0.046964,0.001025,-0.019347],
            [0.064189,0.002580,-0.004163],
            [0.045812,0.000935,0.011155],
            [0.014475,-0.022390,0.035821],
            [-0.030356,-0.021950,0.032867],
            [-0.056451,-0.009969,0.002427],
            [-0.056799,-0.010344,-0.010811],
            [-0.029535,-0.022752,-0.040288],
            [0.014879,-0.026222,-0.044973],
            # [0.049378,0.009013,-0.003857],
            [0.041289,0.022323,-0.018326],
        ]
    ),
    '0607': np.array(
        [
            [-0.000719,0.028214,0.002882],
            [-0.020009,0.025397,0.014843],
            [-0.011605,0.027249,-0.013229],
            [0.012941,0.025350,-0.006407],
            [0.003434,0.025193,0.008526],
        ]
    ),
    '0639': np.array(
        [
            [-0.018955,0.060936,0.015469],
            [-0.020800,-0.023902,-0.000520],
            [-0.030419,-0.054724,-0.002630],
            [-0.036102,-0.070761,-0.002145],
            [-0.011164,-0.055726,-0.020575],
            [0.018232,-0.023257,-0.045483]
        ]
    )
}

selected_frame_dict = {
    '0600': 0, '0604': 200, '0606': 450, "0607": 200, '0639': 200,
}

if __name__ == '__main__':
    args = parse_args()

    # Get all obj names:
    annotate_dir = args.annotate_dir
    os.makedirs(annotate_dir, exist_ok=True)

    obj_names = os.listdir(args.data_base_path)
    id2full_name = {name[:4]: name for name in obj_names if "-" in name}
    for obj_id in args.obj_ids:
        if obj_id not in id2full_name:
            continue
        obj_full_name = id2full_name[obj_id]
        obj_name = obj_full_name.split('-')[1]
        seq_path = osp.join(args.data_base_path, obj_full_name, '-'.join([obj_name, '1']))

        scanned_model_path = osp.join(args.data_base_path, obj_full_name, 'model.ply')
        # scanned_model_path = osp.join(args.data_base_path, obj_full_name, 'model_realigned.ply')
        # scanned_model_path = osp.join(args.data_base_path, obj_full_name, 'model_aligned.ply')
        aligned_model_save_path = osp.join(args.data_base_path, obj_full_name, 'model_realigned.ply')
        img_dir = osp.join(seq_path, 'color')
        intrin_dir = osp.join(seq_path, 'intrin_ba')
        pose_dir = osp.join(seq_path, "poses_ba")

        # img_name = natsort.natsorted(os.listdir(img_dir))[selected_frame_dict[obj_id]]
        img_name = natsort.natsorted(os.listdir(img_dir))[600]
        img_path = osp.join(img_dir, img_name)
        img_id = osp.splitext(img_name)[0]
        intrin = np.loadtxt(osp.join(intrin_dir, img_id + '.txt'))
        pose = np.loadtxt(osp.join(pose_dir, img_id + '.txt'))

        os.system(f"cp {img_path} {osp.join(annotate_dir, '_'.join([obj_id,obj_name]) + '.png')}")
        # Render for the first image
        render_cad_model_to_depth(scanned_model_path, intrin, pose, H=512, W=512, depth_img_save_path=osp.join(args.rendered_save_dir, '_'.join([obj_id,obj_name]) + '.png'), origin_img_path=img_path)

        # # Load annotated 2D points:
        # label_path = osp.join(annotate_dir, '_'.join([obj_id ,obj_name]) + '.json')
        # with open(str(label_path)) as f:
        #     annotations = json.load(f)
        # # Sort by the label
        # annotations = sorted(annotations["shapes"], key=lambda x: str(x["label"]))
        # kpts3D = annotated_3d[obj_id]
        # kpts2D = np.stack(annotations[0]['points'][:len(kpts3D)]) # N * 2

        # pose_new, pose_new_homo, inliers, state = ransac_PnP(intrin, pts_2d=kpts2D, pts_3d=kpts3D, pnp_reprojection_error=5, img_hw=[512,512], use_pycolmap_ransac=True)
        # render_cad_model_to_depth(scanned_model_path, intrin, pose_new_homo, H=512, W=512, depth_img_save_path=osp.join(args.rendered_save_dir, '_'.join([obj_id,obj_name]) + '_annotated.png'), origin_img_path=img_path)

        # # Transform pose:
        # transform_offset = np.linalg.inv(pose) @ pose_new_homo
        # mesh = trimesh.load(scanned_model_path)
        # mesh_transformed = mesh.apply_transform(transform_offset)
        # render_cad_model_to_depth(mesh_transformed, intrin, pose, H=512, W=512, depth_img_save_path=osp.join(args.rendered_save_dir, '_'.join([obj_id,obj_name]) + '_transformed.png'), origin_img_path=img_path)
        # mesh_transformed.export(aligned_model_save_path)