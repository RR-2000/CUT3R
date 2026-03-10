import os
import glob
from tqdm import tqdm

# Define the merged dataset metadata dictionary
dataset_metadata = {
    "davis": {
        "img_path": "/home/ramanathan/data/DAVIS-2017/JPEGImages/Full-Resolution",
        "mask_path": "/home/ramanathan/data/DAVIS-2017/Annotations/Full-Resolution",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,  # Not used in mono depth estimation
    },
    "segTrackv2": {
        "img_path": "/home/ramanathan/data/SegTrackv2/JPEGImages",
        "mask_path": "/home/ramanathan/data/SegTrackv2/JPEGImages",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,  # Not used in mono depth estimation
    },
    "FBMS-56": {
        "img_path": "/home/ramanathan/data/FBMS-56/Large_vids",
        "mask_path": "/home/ramanathan/data/FBMS-56/Large_vids",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,  # Not used in mono depth estimation
    },
    "kubric-custom": {
        "img_path": "/mnt/rdata4_5/kubric_custom/test",
        "mask_path": "/mnt/rdata4_5/kubric_custom/test",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,  # Not used in mono depth estimation
    },
}