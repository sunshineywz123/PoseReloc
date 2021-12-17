from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os 
import os.path as osp
import math
import matplotlib.cm as cm
import cv2
from einops import repeat
from kornia.utils.grid import create_meshgrid
from src.utils.utils import plot_image_pair, plot_keypoints, plot_matches

jet = cm.get_cmap("jet")  # "Reds"
jet_colors = jet(np.arange(256))[:, :3]  # color list: normalized to [0,1]

def blend_img_heatmap(img, heatmap, alpha=0.5):  # (H, W, *)  # (H, W, 3)
    h, w = heatmap.shape[:2]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2:
        img = repeat(img, "h w -> h w c", c=3)
    img = np.uint8(img)
    heatmap = np.uint8(heatmap * 255)
    blended = np.asarray(
        Image.blend(Image.fromarray(img), Image.fromarray(heatmap), alpha)
    )
    return blended

def draw_matches(data_dict, match_dict, save_path=None):
    # Load images
    image_path0 = osp.join(data_dict['image_base_dir'], data_dict["img_name0"])
    image_path1 = osp.join(data_dict['image_base_dir'], data_dict["img_name1"])
    image0 = np.asarray(Image.open(image_path0))
    image1 = np.asarray(Image.open(image_path1))

    plot_image_pair([image0, image1])
    plot_matches(match_dict['mkpts0_f'], match_dict['mkpts1_f'], color = np.zeros((match_dict['mkpts1_f'].shape[0], 3)))

    # plot distance maps!

    # Save plot
    if save_path is not None:
        os.makedirs(save_path.rsplit('/', 1)[0], exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()

def draw_local_heatmaps(data_dict, distance_map, center_location, save_path=None):
    """
    distance_map: N*ww*1
    """
    image_path1 = osp.join(data_dict['image_base_dir'], data_dict["img_name1"])
    image1 = np.asarray(Image.open(image_path1))
    h, w = image1.shape[:2]

    M, WW, C = distance_map.shape
    W = int(math.sqrt(WW))
    distance_map_max = np.max(distance_map, axis=-2, keepdims=True)
    distance_map_min = np.min(distance_map, axis=-2, keepdims=True)
    distance_normalized = (distance_map - distance_map_min) / (distance_map_max - distance_map_min + 1e-6) 
    distance_normalized = np.reshape(distance_normalized, (-1, W,W)) # L*W*W

    grid = (create_meshgrid(W,W, normalized_coordinates=False,) - W // 2 ).cpu().numpy()
    coordinate_grid = (center_location[:, None, None, :] + grid).astype(np.int) # L*W*W*2

    response_map = np.zeros((h,w))
    response_map[coordinate_grid[..., 1], coordinate_grid[..., 0]] = distance_normalized # h*w
    response_map = np.uint8(255 * response_map)
    colored_response_map = jet_colors[response_map] # h*w*3

    img_blend_with_heatmap = blend_img_heatmap(image1, colored_response_map, alpha=0.5)

    if save_path is not None:
        os.makedirs(save_path.rsplit('/', 1)[0], exist_ok=True)
        Image.fromarray(img_blend_with_heatmap).save(save_path)

def visualize_colmap_3D(images, cameras, point3Ds, image_path, save_path):
    assert osp.exists(image_path)
    os.makedirs(save_path, exist_ok=True)

    for id, image in images:
        img_name = image.name
    for i, image_path in enumerate(img_paths):
        # Load image and keypoints
        im, _ = load_image(
            image_path, use_color_image=True, crop_center=False, force_rgb=True
        )
        used = None
        key = os.path.splitext(os.path.basename(image_path))[0]
        if best_index != -1:
            for j in colmap_images:
                if key in colmap_images[j].name:
                    # plot all keypoints
                    used = colmap_images[j].point3D_ids != -1
                    registed_points_mask.append(used)
                    registed_index.append(i)
                    break
        if used is None:
            used = [False] * keypoints_dict[key].shape[0]
        used = np.array(used)

        fig = plt.figure(figsize=(20, 20))
        plt.imshow(im)
        plt.plot(
            keypoints_dict[key][~used, 0],
            keypoints_dict[key][~used, 1],
            "r.",
            markersize=12,
        )
        plt.plot(
            keypoints_dict[key][used, 0],
            keypoints_dict[key][used, 1],
            "b.",
            markersize=12,
        )
        plt.tight_layout()
        plt.axis("off")

        # TODO Ideally we would save to pdf
        # but it does not work on 16.04, so we do png instead
        # https://bugs.launchpad.net/ubuntu/+source/imagemagick/+bug/1796563
        viz_file_hq = os.path.join(
            viz_folder_hq,
            "image{:02d}_yes.png".format(i)
            if i in registed_index
            else "image{:02d}_no.png".format(i),
        )
        viz_file_lq = os.path.join(
            viz_folder_lq,
            "image{:02d}_yes.jpg".format(i)
            if i in registed_index
            else "image{:02d}_no.png".format(i),
        )
        plt.savefig(viz_file_hq, bbox_inches="tight")

        # Convert with imagemagick
        os.system(
            'convert -quality 75 -resize "640>" {} {}'.format(viz_file_hq, viz_file_lq)
        )

        plt.close()

    print(
        f"{len(img_paths)} images in bag, index: {np.array(registed_index)} registrated"
    )
    for i, mask in enumerate(registed_points_mask):
        print(f"\nindex :{registed_index[i]}, {mask.sum()}/{len(mask)} |")

    print("Done [{:.02f} s.]".format(time() - t_start))
