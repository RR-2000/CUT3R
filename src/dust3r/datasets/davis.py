import os.path as osp
import cv2
import PIL
import numpy as np
import itertools
import os
import sys
import glob

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from scipy.interpolate import griddata
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates


class DAVIS(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        self.version = kwargs.pop("version", "all")
        super().__init__(*args, **kwargs)

        if self.split == "test":
            self.split = "val"

        assert self.split in ["train", "val"]
        assert self.version in ["all", "2016", "2017"]
        self.loaded_data = self._load_data(self.split, self.version)

    def _load_data(self, split, version):
        root = self.ROOT

        with open(os.path.join(root, "ImageSets", "2016", f"{split}.txt"), "r") as f:
            scenes_2016 = [line.strip() for line in f.readlines()]
        with open(os.path.join(root, "ImageSets", "2017", f"{split}.txt"), "r") as f:
            scenes_2017 = [line.strip() for line in f.readlines()]

        if version == "2016":
            scenes_to_use = scenes_2016
        elif version == "2017":
            scenes_to_use = scenes_2017
        else: # All
            scenes_to_use = list(set(scenes_2016 + scenes_2017))
        
        self.scenes = scenes_to_use

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes, desc=f"Loading {split} scenes from DAVIS-{version} dataset"):
            scene_dir = osp.join(root,'JPEGImages','Full-Resolution', scene)

            rgb_dir = scene_dir
            basenames = sorted(
                [f.split('_')[-1][:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
            )
            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            start_img_ids_ = img_ids[0]

            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue

            start_img_ids.extend([start_img_ids_])
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
            scene = self.scenes[scene_id]
            scene_dir = osp.join(self.ROOT,'JPEGImages','Full-Resolution', scene)
            rgb_dir = scene_dir
            dymask_dir = scene_dir.replace('JPEGImages', 'Annotations')
            # depth_dir = osp.join(scene_dir, "depths")
            # cam_pose_dir = osp.join(scene_dir, "extrinsics")
            # cam_intr_dir = osp.join(scene_dir, "intrinsics")


            basename = self.images[view_idx]

            # Load RGB image

            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            # rgb_image = PIL.Image.fromarray(imread_cv2(osp.join(rgb_dir, basename + ".jpg")))

            # Load depthmap
            # depthmap = cv2.imread(osp.join(depth_dir, "depth_" + basename + ".png"), cv2.IMREAD_ANYDEPTH)
            # depthmap = depthmap.astype(np.float32) / 65535.0 * 1000.0
            # depthmap[~np.isfinite(depthmap)] = 0  # invalid
            # depthmap[depthmap > 1000] = 0.0

            # cam = np.load(osp.join(cam_dir, basename + ".npz"))
            # camera_pose = np.load(osp.join(cam_pose_dir, "extrinsic_" + basename + ".npy"))
            # intrinsics = np.load(osp.join(cam_intr_dir, "intrinsic_" + basename + ".npy"))
            # rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            #     rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            # )

            # # generate img mask and raymap mask
            # img_mask, ray_mask = self.get_img_and_ray_masks(
            #     self.is_metric, v, rng, p=[0.9, 0.05, 0.05]
            # )

            # # 3D points and valid mask
            # pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
            #     depthmap, intrinsics, camera_pose
            # )

            # load dynamic mask
            dynamic_mask = imread_cv2(osp.join(dymask_dir, basename + ".png"), cv2.IMREAD_GRAYSCALE)
            dynamic_mask = (dynamic_mask > 0).astype(np.int32)

            # Fake instrinsics
            intrinsics = np.array([[1.0, 0.0, dynamic_mask.shape[-1] / 2.0],
                                   [0.0, 1.0, dynamic_mask.shape[-2] / 2.0],
                                   [0.0, 0.0, 1.0]], dtype=np.float32)

            rgb_image, dynamic_mask, intrinsics = self._crop_resize_if_necessary(
                rgb_image, dynamic_mask, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(dict(
                    img=rgb_image,
                    dynamic_mask=dynamic_mask,
                    depthmap=dynamic_mask,  # Placeholder since depthmap is irrelevant here
                    camera_intrinsics=np.identity(3).astype(np.float32),  # Placeholder since intrinsics are not loaded
                    camera_pose=np.identity(4).astype(np.float32),  # Placeholder since camera pose is not loaded
                    img_mask=True,  # Placeholder since img_mask is not generated
                    ray_mask=False,  # Placeholder since ray_mask is not generated
                    dataset=f"Davis-{self.version}",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".jpg"),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
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
        default='/home/ramanathan/data/DAVIS-2017',
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

    dataset = DAVIS(
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