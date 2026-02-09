import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.dymask.metadata import metadata as dymask_dataset_metadata
from eval.dymask.utils import save_dymask_masks
from accelerate import PartialState
from dust3r.datasets import get_data_loader
import time
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset", type=str, default="PointOdyssey", choices=["PointOdyssey", "Davis-16", "Davis-17", "Davis-All"], help="dataset to evaluate",
    ) # TODO: expand choices

    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--size", type=int, default="224")


    return parser


def eval_dymask(args, model, save_dir=None):

    eval_dymask_dist(args, model, save_dir=save_dir)

    return


def eval_dymask_dist(args, model, save_dir=None):
    from dust3r.inference import inference

    data_loader = build_dataset(
        dymask_dataset_metadata[args.eval_dataset],
        1,# batch size
        args.num_workers,
        accelerator=None,
        test=True,
        fixed_length=False
    )

    if save_dir is None:
        save_dir = args.output_dir

    model.eval()
    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    epoch = 0

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)


    with distributed_state.split_between_processes(data_loader) as batches:
        for batch in tqdm(batches):
            outputs, _ = inference(batch, model, device, verbose=False)

            # import pdb; pdb.set_trace()
            gt_dymasks = [gt["dynamic_mask"].squeeze(-1) for gt in batch]
            pred_dymasks = [out["dynamic_mask"].squeeze(-1) for out in outputs['pred']]

            video_save_dir = os.path.join(save_dir, batch[0]["label"][0]+batch[0]["rng"][0].item().__str__())
            os.makedirs(video_save_dir, exist_ok=True)
            # check gt and pred shapes and stats
            if not (len(gt_dymasks) == len(pred_dymasks)):
                print("Mismatch in number of frames between GT and Pred")
            # print(f"GT Dymask Shape: {gt_dymasks[0].shape}, min: {gt_dymasks[0].min().item()}, max: {gt_dymasks[0].max().item()}")
            # print(f"Pred Dymask Shape: {pred_dymasks[0].shape}, min: {pred_dymasks[0].min().item()}, max: {pred_dymasks[0].max().item()}")
            # exit()
            save_dymask_masks(
                pred_dymasks,
                video_save_dir,
                tag="pred_"
            )
            save_dymask_masks(
                gt_dymasks,
                video_save_dir,
                tag="gt_"
            )
            save_dymask_masks(
                pred_dymasks,
                video_save_dir,
                tag="pred_",
                npy=False
            )
            save_dymask_masks(
                gt_dymasks,
                video_save_dir,
                tag="gt_",
                npy=False
            )
            # exit()

    return

def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length
    )
    return loader

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    # from dust3r.utils.image import load_images_for_eval as load_images
    # from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    # from dust3r.utils.camera import pose_encoding_to_camera

    model = ARCroco3DStereo.from_pretrained(args.weights)
    eval_dymask(args, model, save_dir=args.output_dir)
