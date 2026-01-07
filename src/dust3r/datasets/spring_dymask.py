import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2

DEBUG = False


class Spring_DyMask(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 16
        super().__init__(*args, **kwargs)
        self.ROOT = os.path.join(self.ROOT, self.split)

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.scenes = os.listdir(self.ROOT)

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "frame_left")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")]
            )[1:-1] # Remove first and last frames for DyMask dataset
            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            # start_img_ids_ = img_ids[:-self.num_views+1]
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
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

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "frame_left")
            depth_dir = osp.join(scene_dir, "depth")# NOT USED
            cam_dir = osp.join(scene_dir, "cam")# NOT USED
            dymask_dir = osp.join(scene_dir, "maps", "rigidmap_BW_left")

            basename = self.images[view_idx]

            # Load RGB
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))

            # print(osp.join(rgb_dir, basename + ".png"))
            # Load depthmap
            if False:
                depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
                depthmap[~np.isfinite(depthmap)] = 0  # invalid
            else:
                depthmap = np.ones_like(rgb_image[:, :, 0])

            # Load dynamic_mask
            # print(osp.join(dymask_dir, basename.replace("frame_left_", "rigidmap_BW_left_") + ".png"))
            dynamic_mask_raw = imread_cv2(osp.join(dymask_dir, basename.replace("frame_left_", "rigidmap_BW_left_") + ".png"))
            # Resize to Half resolution for consistency
            dynamic_mask_raw = cv2.resize(dynamic_mask_raw, (dynamic_mask_raw.shape[1]//2, dynamic_mask_raw.shape[0]//2), interpolation=cv2.INTER_AREA)

            
            if False:

                cam = np.load(osp.join(cam_dir, basename + ".npz"))
                camera_pose = cam["pose"]
                intrinsics = cam["intrinsics"]
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )
            else:
                camera_pose = np.eye(4)
                intrinsics = np.array(
                    [[1050.0, 0.0, rgb_image.shape[1] / 2],
                     [0.0, 1050.0, rgb_image.shape[0] / 2],
                     [0.0, 0.0, 1.0]]
                )
                rgb_image, _, _ = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                dynamic_mask_raw, depthmap, intrinsics = self._crop_resize_if_necessary(
                    dynamic_mask_raw, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )
                np_dy = np.asarray(dynamic_mask_raw)
                dynamic_mask = np_dy[:, :, 0] > 0 # Convert to binary mask

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
                    dataset="spring_dymask",
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
                    # dynamic_mask_sparse=dynamic_mask.astype(np.float32),
                    # valid_mask_sparse=depthmap.astype(np.float32),
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
        default='/mnt/rdata4_3/dymask_datasets/spring/spring',
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

    dataset = Spring_DyMask(
        ROOT=dataset_location,
        split=dset,
        num_views=N,
        resolution=resolution,
        seed = 50,
    )

    for idx in range(len(dataset)):
        views = dataset[idx]
        print(f"Scene {idx} has {len(views)} views")
        for key, value in views[0].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}, {value.dtype}")
            else:
                print(f"  {key}: {type(value)}, {value}")
            exit()
