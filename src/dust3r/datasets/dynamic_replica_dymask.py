import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys
import torch

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


DEBUG = False

class DynamicReplica_DyMask(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 16
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        self.scenes = [scene for scene in os.listdir(os.path.join(self.ROOT, split)) if scene.endswith("left")]

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.ROOT, self.split, scene)
            rgb_dir = osp.join(scene_dir, "images")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")],
                key=lambda x: float(x.split("-")[-1]),
            )
            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue

            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
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

    def _get_dymask(self, id_mask, trajectories_3d, view_id):

        pass

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

        # Load trajectories ONCE for all views to ensure id_mask is available
        scene_id = self.sceneids[image_idxs[0]]
        scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
        traj_dir = osp.join(scene_dir, "trajectories")
        
        try:
            traj_3d_paths = [os.path.join(traj_dir, f'{int(idx.split("-")[-1]):06d}.pth') for idx in image_basenames]
            trajs_3d_full = [torch.load(traj_3d_paths[i], map_location='cpu', weights_only=False) for i in range(len(image_basenames))]
            trajs_3d = [traj['traj_3d_world'].numpy() for traj in trajs_3d_full]
            trajs_3d_stack = np.stack(trajs_3d, axis=0)
            trajs_ids = np.stack([traj['instances'].numpy() for traj in trajs_3d_full], axis=0)

            displacements = (trajs_3d_stack - trajs_3d_stack[0])  # (T, N, 3)
            displacement_magnitudes = np.linalg.norm(displacements, axis=-1)  # (T, N)
            movement_threshold = 0.0001  # Define a threshold for movement (from SegAnyMo)
            moving_points = (np.max(displacement_magnitudes, axis=0) > movement_threshold)
            
            d_instance = trajs_ids[:, moving_points]  # (T, M) M: number of moving points
            # Determine whether an obj is dynamic
            total_counts = np.bincount(trajs_ids.reshape(-1).astype(np.int64))
            counts = np.bincount(d_instance.reshape(-1).astype(np.int64), minlength=total_counts.shape[0])
                
            half_total_counts = total_counts / 2
            id_mask = (counts > half_total_counts)[total_counts != 0]

            # Delete large arrays to save memory
            del trajs_3d_stack, trajs_ids, displacements, displacement_magnitudes, counts, total_counts, half_total_counts
            del trajs_3d_full, trajs_3d, traj_3d_paths, moving_points, d_instance
        except Exception as e:
            # Fallback: if trajectories fail to load, use empty mask
            print(f"Warning: Failed to load trajectories for scene {scene_id}, sample {idx}: {e}")
            id_mask = np.array([], dtype=bool)

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "images")
            depth_dir = osp.join(scene_dir, "depths")
            mask_id_dir = osp.join(scene_dir, "instance_id_maps")
            cam_dir = osp.join(self.ROOT, "cams", self.scenes[scene_id], "camera") 

            basename = image_basenames[v]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))

            # Load depthmap
            depthmap = cv2.imread(osp.join(depth_dir, basename.replace('left-', 'left_') + ".geometric.png"), cv2.IMREAD_ANYDEPTH)
            depthmap = depthmap.astype(np.float32) / 65535.0 * 1000.0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depthmap[depthmap > 1000] = 0.0 # Cap depth to 1000m

            # Load dynamic mask
            dynamic_mask_raw = imread_cv2(osp.join(mask_id_dir, basename.replace('left-', 'left_') + ".png"), cv2.IMREAD_ANYDEPTH)
            np_dy = np.asarray(dynamic_mask_raw)
            if id_mask.size > 0:  # Only apply mask if trajectories were loaded successfully
                np_dy[np.isin(np_dy, np.where(id_mask)[0])] = 1
            else:
                np_dy[:] = 0  # No dynamic objects if trajectories unavailable
            

            if False:

                cam = np.load(osp.join(cam_dir, basename.split('-')[-1] + ".npz"))
                camera_pose = cam["pose"]
                intrinsics = cam["intrinsics"]
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )
            else:
                cam = np.load(osp.join(cam_dir, basename.split('-')[-1] + ".npz"))
                camera_pose = cam["pose"]
                intrinsics = cam["intrinsics"]

                # camera_pose = np.eye(4)
                # intrinsics = np.array(
                #     [[1050.0, 0.0, rgb_image.shape[1] / 2],
                #      [0.0, 1050.0, rgb_image.shape[0] / 2],
                #      [0.0, 0.0, 1.0]]
                # )
                rgb_image, _, _ = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                dynamic_mask_raw, depthmap, intrinsics = self._crop_resize_if_necessary(
                    dynamic_mask_raw, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )
                np_dy = np.asarray(dynamic_mask_raw)
                dynamic_mask = np_dy > 0 # Convert to binary mask

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.85, 0.10, 0.05]
            )

            if DEBUG:
                # convert from PIL to numpy
                np_rgb_image = np.asarray(rgb_image)
                print(np_rgb_image.shape, dynamic_mask.shape)
                
                cv2.imwrite(f"/home/ramanathan/Methods/CUT3R/tmp/rgb_{v}.png", np_rgb_image[:, :, ::-1].astype(np.uint8))
                cv2.imwrite(f"/home/ramanathan/Methods/CUT3R/tmp/dymask_{v}.png", (dynamic_mask * 255).astype(np.uint8))

                # Overlayed image
                overlay = np_rgb_image.copy()
                overlay[dynamic_mask] = [0, 0, 255]  # Mark dynamic regions in red
                cv2.imwrite(f"/home/ramanathan/Methods/CUT3R/tmp/overlay_{v}.png", overlay[:, :, ::-1].astype(np.uint8))


            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    dynamic_mask=dynamic_mask.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="dynamic_replica_dymask",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
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
        default='/mnt/rdata4_3/dymask_datasets/DynamicReplica',
        help="path to dataset",
    )
    args = parser.parse_args()

    dataset_location = args.dataset_location
    dset = "train"
    use_augs = False
    S = 2
    N = 16
    resolution = [(512, 288)]
    strides = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    clip_step = 2
    quick = True  # Set to True for quick testing

    dataset = DynamicReplica_DyMask(
        ROOT=dataset_location,
        split=dset,
        num_views=N,
        resolution=resolution
    )

    for idx in range(len(dataset)):
        views = dataset[idx]
        print(f"Scene {idx} has {len(views)} views")
        for key, value in views[0].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}, {value.dtype}")
            else:
                print(f"  {key}: {type(value)}, {value}")