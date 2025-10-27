import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.dymask.tools import dymask_evaluation, group_by_directory
import numpy as np
import cv2
from tqdm import tqdm
import glob
import warnings
from PIL import Image
import argparse
import json


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="PointOdyssey", choices=["PointOdyssey", "Davis-16", "Davis-17", "Davis-All"], help="dataset to evaluate",
    )
    parser.add_argument(
        "--border_th",
        type=float,
        default=0.008,
        help="boundary threshold for boundary F-measure computation",
        )

    parser.add_argument('--conf_th', type=float, default=0.5, help='confidence threshold to binarize dynamic masks')
    return parser


#### Taken and adapted from davis2017/utils.py ####

def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D

def main(args):
    if args.eval_dataset in ["PointOdyssey", "Davis-16", "Davis-17", "Davis-All"]:

        def mask_read(filename):
            """Read mask data from file, return as numpy array."""
            mask = np.load(filename) > args.conf_th
            return mask

        pred_paths = glob.glob(
            f"{args.output_dir}/*/pred_frame_*.npy"
        )
        pred_paths = sorted(pred_paths)

        gt_paths = glob.glob(
            f"{args.output_dir}/*/gt_frame_*.npy"
        )
        gt_paths = sorted(gt_paths)

        def get_video_results():
            
            # import pdb; pdb.set_trace()
            grouped_pred_dymask = group_by_directory(pred_paths)

            grouped_gt_dymask = group_by_directory(gt_paths)
            gathered_dymask_metrics = []

            for key in tqdm(grouped_pred_dymask.keys()):
                pd_pathes = grouped_pred_dymask[key]
                gt_pathes = grouped_gt_dymask[key]

                gt_dymask = np.concatenate(
                    [mask_read(gt_path) for gt_path in gt_pathes], axis=0
                )
                pred_dymask = np.concatenate(
                    [mask_read(pd_path) for pd_path in pd_pathes], axis=0
                )
                dymask_results = (
                    dymask_evaluation(
                        pred_dymask,
                        gt_dymask,
                        bound_th=args.border_th,
                    )
                )
                JM,JR,JD = db_statistics(dymask_results["J"])
                FM,FR,FD = db_statistics(dymask_results["F"])
                gathered_dymask_metrics.append({'JM': JM, 'JR': JR, 'JD': JD, 'FM': FM, 'FR': FR, 'FD': FD})

            dymask_log_path = f"{args.output_dir}/result_{args.border_th}.json"
            # Get the median, mean and std of the evaluation metrics
            average_metrics = {}
            for metric in gathered_dymask_metrics[0].keys():
                metric_values = [
                    video_metrics[metric] for video_metrics in gathered_dymask_metrics
                ]
                average_metrics[metric] = {
                    "mean": float(np.mean(metric_values)),
                    "std": float(np.std(metric_values)),
                    "median": float(np.median(metric_values)),
                }
            print("Average dymask evaluation metrics:", average_metrics)
            with open(dymask_log_path, "w") as f:
                f.write(json.dumps(average_metrics))

        get_video_results()
    else:
        raise NotImplementedError(
            f"Dymask evaluation for {args.eval_dataset} is not implemented yet."
        )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
