import os.path as osp
import random
import cv2
import numpy as np
import itertools
import os
import sys
import glob
import math
import pandas as pd
import json

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from scipy.interpolate import griddata
import imageio.v2 as iio
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates


class Stereo4D_Multiview(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.sample_save = kwargs.pop('sample_save', False)
        self.max_interval = 4
        super().__init__(*args, **kwargs)
        assert self.split in ["train"] # only train split available
        self.loaded_data = self._load_data(self.split)

    def largest_displacement(self, tracks):
        """Compute the largest displacement of 3D tracks over time.

        Args:
            tracks (np.ndarray): Array of shape (num_points, num_frames, 3) representing 3D tracks.

        Returns:
            float: The largest displacement observed in the tracks.
        """
        # Find the first valid point for each track
        valid_mask = np.isfinite(tracks).all(axis=2)  # (num_points, num_frames)
        first_valid_indices = np.argmax(valid_mask, axis=1)  # (num_points,)
        first_valid_points = tracks[np.arange(tracks.shape[0]), first_valid_indices]  # (num_points, 3)
        # Compute displacements from the first valid point
        displacements = np.linalg.norm(tracks - first_valid_points[:, np.newaxis, :], axis=2)  # (num_points, num_frames)
        max_displacement = np.nanmax(displacements)
        return max_displacement
        
    def _load_data(self, split):
        root = os.path.join(self.ROOT)
        self.scenes = []

        # Read dataframe of valid scenes
        df = pd.read_csv("/home/ramanathan/data/pos_stereo4d_results.csv")
        # Only keep scenes with pos_difference_max > 1m
        # valid_scenes = set(df['name'][df['max_distance'] > 1.0].tolist())
        valid_scenes = []
        for scene in tqdm(random.sample(df['name'].tolist(), len(df['name'].tolist())), desc="Filtering valid scenes"):
            meta_file = osp.join(root, scene, f'{scene}_rectified_perspective.npz')
            if not osp.isfile(meta_file):
                print(f"Meta file not found for {scene}, skipping.")
                continue
            tracks = np.load(meta_file)['tracks_3d']  # (num_points, num_frames, 3)
            max_disp = self.largest_displacement(tracks)
            if max_disp > 10.0 and max_disp < 50.0:
                valid_scenes.append(scene)

            print(f"Valid scene found: {scene}, max displacement: {max_disp:.3f}m")
            print(len(valid_scenes))
            if len(valid_scenes) >= 50:
                break
            

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene_dir in tqdm(glob.glob(os.path.join(root, "*/"))):
            scene = os.path.basename(os.path.normpath(scene_dir))
            if scene not in valid_scenes:
                print(f"Skipping {scene} as it is not in valid scenes.")
                continue

            # Check if meta file exists
            meta_file = osp.join(scene_dir, f'{scene}_rectified_perspective.npz')
            if not osp.isfile(meta_file):
                print(f"Meta file not found for {scene}, skipping.")
                continue
            rgb_dir = scene_dir
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")]
            )
            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            # start_img_ids_ = img_ids[:-self.num_views+1]

            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue

            start_img_ids.extend(start_img_ids_)
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            # offset groups
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=1.0,
        )
        image_idxs = np.array(all_image_ids)[pos]
        image_basenames = [self.images[i] for i in image_idxs]
        meta = np.load(osp.join(self.ROOT, self.scenes[self.sceneids[image_idxs[0]]], 
            f'{self.scenes[self.sceneids[image_idxs[0]]]}_rectified_perspective.npz'))
        tracks = meta['tracks_3d']  # (num_points, num_frames, 3)
        times = [int(basename.split('_')[-1]) for basename in image_basenames]

        # Get motion magnitudes handling NaN values
        diff = tracks[:, 1:] - tracks[:, :-1]  # (num_points, num_frames-1, 3)
        valid_motion = np.isfinite(diff).all(axis=2)  # (num_points, num_frames-1)
        motion_magnitudes = np.zeros_like(diff[..., 0])  # (num_points, num_frames-1)
        motion_magnitudes[valid_motion] = np.linalg.norm(diff[valid_motion], axis=-1)
        # Threshold to get dynamic points
        # dynamic_threshold = 0.01  # You can adjust this threshold
        # dynamic_mask_per_frame = motion_magnitudes > dynamic_threshold  # (num_points, num_frames-1)
        # Use a percentile-based threshold to determine dynamic points
        # dynamic_thresholds = np.percentile(motion_magnitudes[valid_motion], 75, axis=0)  # (num_frames-1,)
        # dynamic_mask_per_frame = motion_magnitudes > dynamic_thresholds  # (num_points, num_frames-1)

        # # Use a percentile-based threshold to determine static points
        # static_thresholds = np.percentile(motion_magnitudes[valid_motion], 25, axis=0)  # (num_frames-1,)
        # static_mask_per_frame = motion_magnitudes < static_thresholds  # (num_points, num_frames-1)

        # Combine masks across frames to get overall dynamic mask
        # compute mean motion per point using only valid frames as weights
        valid_counts = valid_motion.sum(axis=1)


        valid_mask = np.isfinite(tracks[:, times,:]).all(axis=2)  # (num_points, num_frames)
        first_valid_indices = np.argmax(valid_mask, axis=1)  # (num_points,)
        first_valid_points = tracks[np.arange(tracks.shape[0]), first_valid_indices]  # (num_points, 3)
        # Compute displacements from the first valid point
        displacements = np.linalg.norm(tracks - first_valid_points[:, np.newaxis, :], axis=2)  # (num_points, num_frames)

        motion_magnitudes = np.nanmax(displacements, axis=-1)  # (num_points,)
        # motion_magnitudes = np.divide(
        #     motion_magnitudes.sum(axis=1),
        #     valid_counts,
        #     out=np.zeros_like(valid_counts, dtype=motion_magnitudes.dtype),
        #     where=valid_counts != 0,
        # )
        
        # only mark the top 75% as dynamic points
        is_dynamic = motion_magnitudes > 0.1#np.percentile(is_static, 75)
        
        # only mark the bottom 25% as static points
        is_static = motion_magnitudes < 0.01#np.percentile(is_static, 25)
        

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            rgb_dir = scene_dir
            depth_dir = None
            cam_pose_dir = scene_dir
            cam_intr_dir = None


            basename = self.images[view_idx]
            time = times[v]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
            rgb_image_og = rgb_image.copy()
            # Load depthmap
            depthmap = cv2.imread(osp.join(rgb_dir, basename + ".png"), cv2.IMREAD_GRAYSCALE)
            width = depthmap.shape[1]
            height = depthmap.shape[0]

            # cam = np.load(osp.join(cam_dir, basename + ".npz"))
            extrinsics = meta['camera_extrinsics'][time]
            R = extrinsics[:3,:3]
            t = extrinsics[:3,3]
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t

            cx = 0.5
            cy = 0.5
            FOV_H = 60.0
            FOV_V = FOV_H * (height / width)
            fx = (1 / 2.0) / math.tan(math.radians(FOV_H / 2.0))
            fy = (1 / 2.0) / math.tan(math.radians(FOV_V / 2.0))

            intrinsics_norm = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]])
            intrinsics = intrinsics_norm.copy()
            intrinsics[0,:] *= width
            intrinsics[1,:] *= height
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.9, 0.05, 0.05]
            )

            # 3D points and valid mask, Sparse GT
            # Project points to image to get valid mask
            tracks_3d_v = tracks[:, time].reshape(-1, 3)
            # Keep only visible points (non NaN in tracks_3d_v)
            valid_3d_mask = np.isfinite(tracks_3d_v).all(axis=1)

            # static thresholding as bottom 25% and dynamic as top 25%
            static_threshold = 0.05 #max(np.percentile(motion_magnitudes[valid_3d_mask], 25), 0.01)
            is_static = motion_magnitudes < static_threshold
            dynamic_threshold = 0.1 #min(np.percentile(motion_magnitudes[valid_3d_mask], 75), 0.05)
            is_dynamic = motion_magnitudes > dynamic_threshold

            # Depth-based thrsholding
            Depth_min = 0.1
            Depth_max = 30.0


            valid_3d_mask = valid_3d_mask & (is_static | is_dynamic)  # also keep static points
            tracks_3d_v = tracks_3d_v[valid_3d_mask]
            dynamic_mask_3d = is_dynamic[valid_3d_mask]
            proj = (intrinsics @ (R @ tracks_3d_v.T + t[:, np.newaxis])).T  # Nx3
            valid_z = (np.abs(proj[:, 2]) > 1e-8) & (proj[:, 2] > Depth_min) & (proj[:, 2] < Depth_max)
            proj_xy = np.zeros_like(proj[:, :2])
            proj_xy[valid_z] = proj[valid_z, :2] / proj[valid_z, 2:3]
            h, w = depthmap.shape
            proj_x_int = np.round(proj_xy[:, 0]).astype(np.int32)
            proj_y_int = np.round(proj_xy[:, 1]).astype(np.int32)
            valid_proj_mask = (
                (proj_x_int >= 0)
                & (proj_x_int < w)
                & (proj_y_int >= 0)
                & (proj_y_int < h)
            )
            # assert np.sum(valid_proj_mask) > 0, f"No valid projections for view {v} in scene {self.scenes[scene_id]}"
            assert len(valid_proj_mask) == len(tracks_3d_v) and len(valid_proj_mask) == len(dynamic_mask_3d), f'Mismatch in lengths for view {v} in scene {self.scenes[scene_id]}'
            
            proj_x_int = proj_x_int[valid_proj_mask]
            proj_y_int = proj_y_int[valid_proj_mask]
            tracks_3d_v = tracks_3d_v[valid_proj_mask]
            motion_mag_valid = motion_magnitudes[valid_3d_mask][valid_proj_mask]

            # Get valid pixels based on projected points
            valid_mask_sparse = np.zeros_like(depthmap, dtype=bool)
            valid_mask_sparse[proj_y_int, proj_x_int] = True

            # Create a sparse dynamic mask
            dynamic_mask_sparse = np.zeros_like(depthmap, dtype=np.float32)
            dynamic_mask_sparse[proj_y_int, proj_x_int] = dynamic_mask_3d[valid_proj_mask]

            # Check percentage of dynamic points in sparse mask
            # num_dynamic_points = np.sum(dynamic_mask_3d[valid_proj_mask])
            # num_total_points = len(dynamic_mask_3d[valid_proj_mask])
            # perc_dynamic = (num_dynamic_points / num_total_points) * 100.0 if num_total_points > 0 else 0.0

            # if perc_dynamic > 30.0:
            #     dynamic_mask_sparse *= 0.0  # Remove dynamic mask if too many dynamic points

            # Color Scale the dynamic mask by motion magnitude for visualization
            # Get an RGB color based on motion magnitude

            # dynamic_mask_sparse_scale = np.zeros_like(np.array(rgb_image), dtype=np.float32)
            # # Normalize motion magnitudes to [0,1]
            # motion_mag_norm = motion_mag_valid / (motion_mag_valid.max() + 1e-8)
            
            # # Create RGB colors - use jet colormap (blue->cyan->yellow->red)
            # colors = np.zeros((len(motion_mag_norm), 3))
            # colors[:,0] = np.clip(4*motion_mag_norm - 2, 0, 1)  # Red
            # colors[:,1] = np.clip(4*np.abs(motion_mag_norm - 0.5) - 1, 0, 1)  # Green
            # colors[:,2] = np.clip(-4*motion_mag_norm + 2, 0, 1)  # Blue


            # # Assign colors to sparse mask
            # dynamic_mask_sparse_scale[proj_y_int, proj_x_int] = colors

            if self.sample_save:
                
                # save img, depth, dynamic mask as images for visualization
                os.makedirs(f'./sample_out/{self.scenes[scene_id]}', exist_ok=True)

                # load meta json if available
                clip_data = {}
                if osp.exists( f'./sample_out/{self.scenes[scene_id]}_clip_data.json'):
                    clip_data = json.load(open(f'./sample_out/{self.scenes[scene_id]}_clip_data.json', 'r'))
                try:
                    clip_data[f'{view_idx}'] = {'min': proj[:, 2:3].min().item(), 'max': proj[:, 2:3].max().item(), 'mean': proj[:, 2:3].mean().item()}
                except:
                    print("Error in computing min/max/mean depth for view ", view_idx)

                # Save the json data
                with open( f'./sample_out/{self.scenes[scene_id]}_clip_data.json', 'w') as f:
                    json.dump(clip_data, f, indent=4)
                
                iio.imwrite(osp.join(f'./sample_out/{self.scenes[scene_id]}', f'rgb_{view_idx:04d}.png'), (np.array(rgb_image)).astype(np.uint8))
                # iio.imwrite(osp.join('./tmp', f'rgb_og_{view_idx:04d}.png'), (np.array(rgb_image_og)).astype(np.uint8))
                # depth_vis = (depthmap / depthmap.max() * 255).astype(np.uint8)
                # iio.imwrite(osp.join('./tmp', f'depth_{view_idx:04d}.png'), depth_vis)
                # dyn_vis = (dynamic_mask_sparse * 255).astype(np.uint8)
                # Save sparse dynamic mask and valid mask
                dyn_sparse_vis = (dynamic_mask_sparse / (dynamic_mask_sparse.max() + 1e-8) * 255).astype(np.uint8)
                iio.imwrite(osp.join(f'./sample_out/{self.scenes[scene_id]}', f'dynmask_sparse_{view_idx:04d}.png'), dyn_sparse_vis)
                # iio.imwrite(osp.join(f'./sample_out/{self.scenes[scene_id]}', f'dynmask_sparse_scale_{view_idx:04d}.png'), (dynamic_mask_sparse_scale * 255).astype(np.uint8))
                iio.imwrite(osp.join(f'./sample_out/{self.scenes[scene_id]}', f'validmask_sparse_{view_idx:04d}.png'), (valid_mask_sparse.astype(np.uint8) * 255).astype(np.uint8))
                # exit()
            views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    dynamic_mask=dynamic_mask_sparse,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="Stereo4D",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".png"),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                    dynamic_mask_sparse=dynamic_mask_sparse,
                    valid_mask_sparse=valid_mask_sparse,
                )
            )
        assert len(views) == num_views
        return views


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_location",
        type=str,
        default='/mnt/rdata4_3/stereo4d/processed',
        help="path to dataset",
    )
    args = parser.parse_args()

    dataset_location = args.dataset_location
    dset = "train"
    use_augs = False
    S = 20
    N = 150
    resolution = [(512, 512)]
    strides = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    clip_step = 2
    quick = False  # Set to True for quick testing

    dataset = Stereo4D_Multiview(
        ROOT=dataset_location,
        split=dset,
        num_views=N,
        resolution=resolution,
        sample_save=True
    )
    
    sampled_idx = random.sample(range(len(dataset)), S if not quick else 2)

    for idx in sampled_idx:
        views = dataset[idx]
        print(f"Scene {idx} has {len(views)} views")
        # for key, value in views[0].items():
            # if isinstance(value, np.ndarray):
            #     print(f"  {key}: {value.shape}, {value.dtype}")
            # else:
            #     print(f"  {key}: {type(value)}, {value}")