import torch
import numpy as np
import cv2
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
import os
from collections import defaultdict


def group_by_directory(pathes, idx=-1):
    """
    Groups the file paths based on the second-to-last directory in their paths.

    Parameters:
    - pathes (list): List of file paths.

    Returns:
    - dict: A dictionary where keys are the second-to-last directory names and values are lists of file paths.
    """
    grouped_pathes = defaultdict(list)

    for path in pathes:
        # Extract the second-to-last directory
        dir_name = os.path.dirname(path).split("/")[idx]
        grouped_pathes[dir_name].append(path)

    return grouped_pathes


def Jaccard_IoU(mask1, mask2):
    """
    Calculate the Jaccard IoU between two binary masks.

    Args:
        mask1 (numpy.ndarray): First binary mask
        mask2 (numpy.ndarray): Second binary mask

    Returns:
        float: Jaccard IoU value
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0  # Both masks are empty
    else:
        return float(intersection) / union

############# Based on davis2017 metrics db_eval_boundary #############

def boundary_eval(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    # import pdb; pdb.set_trace()
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F

def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k as numpy array.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (numpy.ndarray):	Binary boundary map.
    """
    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def dymask_evaluation(
    predicted_mask_original,
    ground_truth_mask_original,
    void_mask_original=None,
    bound_th=0.008
):
    
    boundary_fmeasure = []
    jaccard_iou = []
    for frame_idx in range(ground_truth_mask_original.shape[0]):
        boundary_fmeasure.append(boundary_eval(
            predicted_mask_original[frame_idx],
            ground_truth_mask_original[frame_idx],
            void_mask_original[frame_idx] if void_mask_original is not None else None,
            bound_th
        ))
        jaccard_iou.append(Jaccard_IoU(
            predicted_mask_original[frame_idx],
            ground_truth_mask_original[frame_idx]
        ))
    results = {
        "J": np.array(jaccard_iou),
        "F": np.array(boundary_fmeasure),
    }

    return results