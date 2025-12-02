import os
import glob
import argparse
import cv2
import sys
import numpy as np
from tqdm import tqdm

#!/usr/bin/env python3
"""
combine_samples.py

Read a folder of 3 types of images (via 3 glob patterns), arrange each triple side-by-side,
and write out a video.

Usage:
    python combine_samples.py --dir /path/to/images \
        --patterns "*_input.png" "*_pred.png" "*_gt.png" \
        --output combined.mp4 --fps 10 --height 512 --labels Input Pred GT
"""


def parse_args():
    p = argparse.ArgumentParser(description="Combine 3 image streams side-by-side into a video")
    p.add_argument("--dir", "-d", required=True, help="Directory containing images")
    p.add_argument("--output_dir", default=None, help="Output video directory")
    p.add_argument("--fps", type=int, default=12, help="Frames per second")
    p.add_argument("--height", type=int, default=None, help="Resize each image to this height (preserve aspect ratio). If omitted, uses min height of each triple.")
    return p.parse_args()

def sorted_glob(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    matches.sort()
    return matches

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def combine_images(dymask, img, validmask):
    staticmask = validmask - dymask
    
    # Add staticmask as green overlay on img
    staticmask_colored = cv2.merge([np.zeros_like(staticmask[:,:,0]), staticmask[:,:,0], np.zeros_like(staticmask[:,:,0])])
    img = cv2.addWeighted(img, 1.0, staticmask_colored, 1.0, 0)
    # Add dymask as red overlay on img
    dymask_colored = cv2.merge([np.zeros_like(dymask[:,:,0]), np.zeros_like(dymask[:,:,0]), dymask[:,:,0]])
    img = cv2.addWeighted(img, 1.0, dymask_colored, 1.5, 0)
    return img

def main():
    args = parse_args()

    for sample_dir in tqdm(os.listdir(args.dir), desc="Processing samples"):
        dir = os.path.join(args.dir, sample_dir)
        if args.output_dir is None:
            args.output_dir = args.dir
        os.makedirs(args.output_dir, exist_ok=True)
        output = os.path.join(args.output_dir, f'{sample_dir}.mp4')
        pattern1 = 'dynmask_sparse_*.png'
        pattern2 = 'rgb_*.png'
        pattern3 = 'validmask_sparse_*.png'
        img_paths1 = sorted_glob(dir, pattern1)
        img_paths2 = sorted_glob(dir, pattern2)
        img_paths3 = sorted_glob(dir, pattern3)

        assert len(img_paths1) == len(img_paths2) == len(img_paths3), "Number of images for each pattern must be the same"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        for p1, p2, p3 in zip(img_paths1, img_paths2, img_paths3):
            img1 = read_img(p1)
            img2 = read_img(p2)
            img3 = read_img(p3)

            if args.height is not None:
                target_h = args.height
            else:
                target_h = min(img1.shape[0], img2.shape[0], img3.shape[0])

            img1_resized = resize_to_height(img1, target_h)
            img2_resized = resize_to_height(img2, target_h)
            img3_resized = resize_to_height(img3, target_h)

            combined_img = combine_images(img1_resized, img2_resized, img3_resized)

            if out is None:
                h, w = combined_img.shape[:2]
                out = cv2.VideoWriter(output, fourcc, args.fps, (w, h))

            out.write(combined_img)


if __name__ == "__main__":
    main()