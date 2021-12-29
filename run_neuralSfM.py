import argparse
import os
import os.path as osp
from itertools import combinations


from src.NeuralSfM import neuralSfM


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument(
        "--n_images",
        type=int,
        default=None,
        help="Only process a small subset of all images for debug",
    )
    parser.add_argument("--enable_post_optimization", type=bool, default=True)
    parser.add_argument("--use_ray", action='store_true')

    # Colmap realted
    parser.add_argument("--min_model_size", type=int, default=10)
    parser.add_argument("--filter_max_reproj_error", type=int, default=4)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    work_dir = args.work_dir
    n_images = args.n_images

    # Prepare data structure
    image_pth = osp.join(work_dir, "images")
    assert osp.exists(image_pth), f"{image_pth} is not exist!"
    img_names = sorted(os.listdir(image_pth))
    img_list = [osp.join(image_pth, img_name) for img_name in img_names if '._' not in img_name][:n_images]

    # generate image pairs:
    # exhauctive matching pairs:
    # NOTE: you can add covisible information to generate your own pairs to reduce the matching complexity
    pair_ids = list(combinations(range(len(img_list)), 2))
    img_pairs = []
    for pair_id in pair_ids:
        img_pairs.append(" ".join([img_list[pair_id[0]], img_list[pair_id[1]]]))
    
    colmap_configs = {}
    colmap_configs['min_model_size'] = args.min_model_size
    colmap_configs['filter_max_reproj_error'] = args.filter_max_reproj_error

    neuralSfM(
        img_list,
        img_pairs,
        work_dir=work_dir,
        enable_post_optimization=args.enable_post_optimization,
        use_ray=args.use_ray,
        colmap_configs=colmap_configs
    )


if __name__ == "__main__":
    main()

