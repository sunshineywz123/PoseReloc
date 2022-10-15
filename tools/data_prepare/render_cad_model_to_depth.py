import pyrender
import numpy as np
import trimesh
import os
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "egl"

jet = cm.get_cmap("jet")  # "Reds"
jet_colors = jet(np.arange(256))[:, :3]  # color list: normalized to [0,1]

def depth2color(depth):
    depth_max = np.max(depth)
    depth_min = np.min(depth)
    depth_normalized = (depth - depth_min) / (
        depth_max - depth_min + 1e-6
    )  # to [0,1]
    depth_normalized_jet_color = jet_colors[np.uint8(depth_normalized * 255)]
    depth_color_image = Image.fromarray(np.uint8(depth_normalized_jet_color * 255))
    return depth_color_image

def save_np(data, save_path):
    ext = save_path.split('.')[-1].lower()
    def save_func(data, path):
        if ext == 'npy':
            save = np.save
        elif ext == 'npz':
            save = np.savez
        else:
            raise NotImplementedError(ext)
        with open(path, 'wb') as h:
            save(h, data)

    save_func(data, save_path)

class Renderer:
    """OpenGL mesh renderer

    Used to render depthmaps from a mesh for 2d evaluation
    """

    def __init__(self, height=480, width=640, depth_range_prior=[0.05,1000]):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        assert isinstance(depth_range_prior, list) and depth_range_prior[1] > depth_range_prior[0]
        self.min_depth_prior = depth_range_prior[0]
        self.max_depth_prior = depth_range_prior[1]
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        mesh = pyrender.Mesh.from_trimesh(mesh)
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            znear=self.min_depth_prior,
            zfar=self.max_depth_prior,
        )
        # self.scene.add(cam, pose=pose)
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform
        # return axis_transform @ pose

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


def convert_pose2T(pose):
    """
    pose: np.array[3*4] or [4*4] or [R: 3*3, t: 3*1 or 3]
    """
    if isinstance(pose, list):
        assert len(pose) == 2
        R, t = pose
        t = t[:, None] if len(t.shape) == 1 else t  # 3*1
        return np.concatenate(
            [np.concatenate([R, t], axis=1), [[0, 0, 0, 1]]], axis=0
        )  # 4*4
    else:
        assert pose.shape[1] == 4
        if pose.shape[0] != 4:
            assert pose.shape[0] == 3
            return np.concatenate([pose, [[0, 0, 0, 1]]], axis=0)
        else:
            return pose

def render_cad_model_to_depth(cat_model_path, K, pose, H, W, depth_npy_save_path=None, depth_img_save_path=None, mask_img_save_path=None, depth_range_prior=None, origin_img_path=None):
    """
    pose: np.array[3*4] or [4*4]
    depth_range_prior: [depth_min, depth_max]
    """
    if isinstance(cat_model_path, str):
        mesh = trimesh.load(cat_model_path)
    else:
        mesh = cat_model_path
    renderer = Renderer() if depth_range_prior is None else Renderer(depth_range_prior=depth_range_prior)

    T = convert_pose2T(pose)
    T_inv = np.linalg.inv(T)
    rgb, depth = renderer(H, W, K, T_inv, mesh)
    
    if depth_img_save_path is not None:
        Path(depth_img_save_path).parent.mkdir(parents=True, exist_ok=True)
        depth_color_image = depth2color(depth)
        if origin_img_path is not None:
            origin_img = Image.open(origin_img_path)
            depth_blend_img = Image.blend(origin_img, depth_color_image, 0.4)
            depth_blend_img.save(depth_img_save_path)
        else:
            depth_color_image.save(depth_img_save_path)
    
    if mask_img_save_path is not None:
        mask = np.zeros((depth.shape[0], depth.shape[1]))
        mask[depth!=0] = 1
        mask_color_image = Image.fromarray(np.uint8(mask * 255))
        mask_color_image.save(mask_img_save_path)

    if depth_npy_save_path is not None:
        save_np(depth_npy_save_path)
    
    return depth