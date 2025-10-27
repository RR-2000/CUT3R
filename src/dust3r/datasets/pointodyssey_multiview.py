import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys
import glob

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from scipy.interpolate import griddata
import imageio.v2 as iio
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates


class PointOdyssey_Multiview(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        super().__init__(*args, **kwargs)
        assert self.split in ["train", "test", "val"]
        self.scenes_to_use = [
            None,
        ]# TODO: specify scenes if needed
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        root = os.path.join(self.ROOT, split)
        self.scenes = []

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene_dir in tqdm(glob.glob(os.path.join(root, "*/"))):
            scene = os.path.basename(os.path.normpath(scene_dir))
            # Check if all required folders exist
            if not all(
                os.path.exists(osp.join(scene_dir, subfolder))
                for subfolder in ["rgbs", "depths", "extrinsics", "intrinsics", "trajs_3d"]
            ):
                print(f"Skipping {scene} due to missing folders.")
                continue

            rgb_dir = osp.join(scene_dir, "rgbs")
            basenames = sorted(
                [f.split('_')[-1][:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
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


        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgbs")
            depth_dir = osp.join(scene_dir, "depths")
            cam_pose_dir = osp.join(scene_dir, "extrinsics")
            cam_intr_dir = osp.join(scene_dir, "intrinsics")

            if v == 0:  # Only load trajectories once for the first view
                # load all trajs_3d for dynamic mask computation
                traj_dir = osp.join(scene_dir, "trajs_3d")
                traj_3d_paths = [os.path.join(traj_dir, 'traj_3d_' + idx + '.npy') for idx in image_basenames]
                trajs_3d = [np.load(traj_3d_paths[i], allow_pickle=True) for i in range(len(image_basenames))]
                trajs_3d_stack = np.stack(trajs_3d, axis=0)


            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, "rgb_" + basename + ".jpg"))
            rgb_image_og = rgb_image.copy()
            # Load depthmap
            depthmap = cv2.imread(osp.join(depth_dir, "depth_" + basename + ".png"), cv2.IMREAD_ANYDEPTH)
            depthmap_og = depthmap.copy()
            depthmap = depthmap.astype(np.float32) / 65535.0 * 1000.0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depthmap[depthmap > 1000] = 0.0

            # cam = np.load(osp.join(cam_dir, basename + ".npz"))
            extrinsics = np.load(osp.join(cam_pose_dir, "extrinsic_" + basename + ".npy"))
            R = extrinsics[:3,:3]
            t = extrinsics[:3,3]
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t

            intrinsics = np.load(osp.join(cam_intr_dir, "intrinsic_" + basename + ".npy"))
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.9, 0.05, 0.05]
            )

            # 3D points and valid mask
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                depthmap, intrinsics, camera_pose
            )

            # No need to reshape as pts3d is already in HxWx3
            # compute dynamic mask based on trajs_3d
            is_same = np.all(np.isclose(trajs_3d_stack, trajs_3d_stack[v:v+1]), axis=(0,2))
            dynamic_mask = np.logical_not(is_same)
            
            # print(trajs_3d_stack[v].shape, dynamic_mask.shape, pts3d.shape)
            dynamic_mask = griddata(trajs_3d_stack[v], dynamic_mask, pts3d, method='nearest', fill_value=0).astype(np.float32)
            # print(f"Dynamic mask stats for view {v} of shape {dynamic_mask.shape}: min {dynamic_mask.min()}, max {dynamic_mask.max()}, mean {dynamic_mask.mean()}, sum {dynamic_mask.sum()}")
            dynamic_mask = np.clip(dynamic_mask, 0, 1)
            # # save img, depth, dynamic mask as images for visualization
            # os.makedirs('./tmp', exist_ok=True)
            # iio.imwrite(osp.join('./tmp', f'rgb_{view_idx:04d}.png'), (np.array(rgb_image) * 255).astype(np.uint8))
            # iio.imwrite(osp.join('./tmp', f'rgb_og_{view_idx:04d}.png'), (np.array(rgb_image_og) * 255).astype(np.uint8))
            # depth_vis = (depthmap / depthmap.max() * 255).astype(np.uint8)
            # iio.imwrite(osp.join('./tmp', f'depth_{view_idx:04d}.png'), depth_vis)
            # iio.imwrite(osp.join('./tmp', f'depth_og_{view_idx:04d}.png'), (np.array(depthmap_og)).astype(np.uint8))
            # dyn_vis = (dynamic_mask * 255).astype(np.uint8)
            # iio.imwrite(osp.join('./tmp', f'dynmask_{view_idx:04d}.png'), dyn_vis)
            # print(np.unique(dynamic_mask, return_counts=True))
            # exit()
            views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    dynamic_mask=dynamic_mask,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="PointOdyssey",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".jpg"),
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
        default='/mnt/rdata4_6/kx_data/4d_dataset/point_odyssey',
        help="path to dataset",
    )
    args = parser.parse_args()

    dataset_location = args.dataset_location
    dset = "test"
    use_augs = False
    S = 2
    N = 16
    resolution = [(512, 288)]
    strides = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    clip_step = 2
    quick = True  # Set to True for quick testing

    dataset = PointOdyssey_Multiview(
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