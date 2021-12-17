import argparse
import os
import os.path as osp

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module
from src.post_optimization.post_optimization import post_optimization


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Match and refinement model related parameters
    # TODO: move configs to cfg
    parser.add_argument("--cfg_path", type=str, default="")
    parser.add_argument("--weight_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=666)

    # Data related
    parser.add_argument("--img_resize_max", type=int, default=1920)
    parser.add_argument("--img_resize_min", type=int, default=800)
    parser.add_argument("--df", type=int, default=8, help="divisible factor.")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument(
        "--subset_name", default=None, help="Path to subset images, nbag_id.txt"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="IMC",
        choices=["IMC"],
        help="Choice: [IMC,]",
    )
    parser.add_argument(
        "--data_part_name",
        default=None,
        help="Used to identify different dataset results",
    )
    parser.add_argument("--n_imgs", default=None, help="Used for debug")

    # COLMAP results related:
    parser.add_argument(
        "--colmap_results_load_dir", default=None, help="Get colmap output results"
    )
    parser.add_argument(
        "--colmap_refined_save_dir",
        default=None,
        help="Colmap refined results save dir",
    )
    parser.add_argument("--colmap_best_index", type=int, default=0)

    # Others
    parser.add_argument("--fine_match_debug", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--evaluation", action="store_true")

    args = parser.parse_args()

    # Post process of args
    args.match_type = "loftr_coarse"

    # Get colmap results path and raw match path
    base_dir_part = [args.colmap_results_load_dir]
    base_dir_part.append(args.match_type)
    (
        base_dir_part.append(args.data_part_name)
    ) if args.data_part_name is not None else None
    (
        base_dir_part.append(osp.splitext(args.subset_name)[0])
    ) if args.subset_name is not None else None
    results_load_dir_medium = osp.join(*base_dir_part)
    args.scene_results_base_dir = results_load_dir_medium
    args.colmap_results_load_dir = osp.join(
        results_load_dir_medium, "colmap_output_path", str(args.colmap_best_index)
    )
    args.raw_match_results_dir = osp.join(
        results_load_dir_medium, "raw_matches/raw_matches.h5"
    )

    # Get save path
    base_dir_part = [args.colmap_refined_save_dir]
    base_dir_part.append(args.match_type)
    (
        base_dir_part.append(args.data_part_name)
    ) if args.data_part_name is not None else None
    (
        base_dir_part.append(osp.splitext(args.subset_name)[0])
    ) if args.subset_name is not None else None
    base_dir_part += ["colmap_output_path", str(args.colmap_best_index)]
    args.results_save_dir = osp.join(*base_dir_part)

    return args


def main():
    args = parse_args()

    post_optimization(args)


if __name__ == "__main__":
    main()
