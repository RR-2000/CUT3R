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

class HOI4D_DyMask(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.max_interval = 16
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        assert split == 'train', "Only 'train' split is available for HOI4D_DyMask"

        self.scenes = [scene for scene in os.listdir(os.path.join(self.ROOT)) if os.path.isdir(os.path.join(self.ROOT, scene))]

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.ROOT, self.split, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")],
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
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            mask_dir = osp.join(scene_dir, "mask")
            cam_dir = osp.join(self.ROOT, "cam")

            basename = image_basenames[v]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))

            # Load depthmap
            depthmap = cv2.imread(osp.join(depth_dir,  basename + ".png"), cv2.IMREAD_ANYDEPTH)
            depthmap = depthmap.astype(np.float32) / 10.0 # convert from decimeters to m
            depthmap[depthmap > 1000] = 0.0 # Cap depth to 1000m
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)

            # Load dynamic mask
            ## TODO: Dymask filtering for moving classes
            dynamic_mask_raw = imread_cv2(osp.join(mask_dir, basename + ".png"))
            np_dy = np.asarray(dynamic_mask_raw) > 0  # Convert to binary mask
            

            if False:

                cam = np.load(osp.join(cam_dir, basename.split('-')[-1] + ".npz"))
                camera_pose = cam["pose"]
                intrinsics = cam["intrinsics"]
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )
            else:
                cam = np.load(osp.join(cam_dir, basename + ".npz"))
                camera_pose = cam["extrinsics"]
                # convert from open3d to pytroch3d convention (flip x and y to negative)
                camera_pose[:3, :3] *= np.array([[-1, -1, 1], [-1, -1, 1], [1, 1, 1]])

                # Following the convention in DynamicReplica_DyMask
                R = camera_pose[:3, :3]
                t = camera_pose[:3, 3]
                camera_pose[:3, :3] = R.T
                camera_pose[:3, 3] = -R.T @ t

                intrinsics = cam["intrinsics"]

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
                    dataset="HOI4D_DyMask",
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
        default='/mnt/rdata4_5/HOI4D',
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

    dataset = HOI4D_DyMask(
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