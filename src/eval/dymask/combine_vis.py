import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import imageio
import argparse
import sys

def combine_and_save_video(input_folder, output_path, fps=30, verbose=False):
    if verbose:
        print(f"Processing folder: {input_folder}")
    # Get all ground truth and prediction images
    gt_images = sorted(glob(os.path.join(input_folder, 'gt_*.png')))
    pred_images = sorted(glob(os.path.join(input_folder, 'pred_*.png')))
    
    if not gt_images or not pred_images:
        print("No images found!")
        return
    
    # Read first image to get dimensions
    first_gt = cv2.imread(gt_images[0])
    height, width = first_gt.shape[:2]
    
    # Create list to store combined frames
    frames = []
    
    # Process each pair of images
    for gt_path, pred_path in zip(gt_images, pred_images):
        # Read images
        gt_img = cv2.imread(gt_path)
        pred_img = cv2.imread(pred_path)
        
        # Convert from BGR to RGB
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        
        # Ensure both images are the same size
        pred_img = cv2.resize(pred_img, (width, height))
        
        # Concatenate images horizontally
        combined = np.hstack((gt_img, pred_img))
        
        # Add to frames list
        frames.append(combined)
    
    # Save as GIF using imageio
    output_path = output_path.replace('.mp4', '.gif')
    imageio.mimsave(output_path, frames, fps=fps)
    if verbose:
        print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine ground truth and prediction images into GIFs')
    parser.add_argument('--in_dir', type=str, help='Experiment directory containing subdirectories with images')
    parser.add_argument('--out_dir', type=str, help='Output directory for saving GIFs', default=None)
    args = parser.parse_args()

    if not os.path.exists(args.in_dir):
        print("Usage: python combine_vis.py <input_directory>")
        sys.exit(1)

    base_dir = args.in_dir
    for subdir in tqdm(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, subdir)

        if args.out_dir is None:
            os.makedirs(os.path.join(base_dir, 'videos'), exist_ok=True)
            output_path = os.path.join(base_dir, 'videos', f"{subdir}.mp4")
        else:
            os.makedirs(args.out_dir, exist_ok=True)
            output_path = os.path.join(args.out_dir, f"{subdir}.mp4")
        
        if os.path.isdir(full_path):
            combine_and_save_video(full_path, output_path, verbose=False)