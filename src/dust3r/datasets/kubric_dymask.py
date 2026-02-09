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


class Kubric_DyMask(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 2
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
            rgb_dir = scene_dir
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.startswith("frame_") and f.endswith(".png")]
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
            
            rgb_dir = scene_dir
            depth_dir = scene_dir
            cam_dir = scene_dir
            dymask_dir = scene_dir

            if v == 0:
                camera_params = np.load(osp.join(cam_dir, "camera_params.npz"))
                object_info = np.load(osp.join(scene_dir, "object_poses.npz"))

            basename = self.images[view_idx]
            frame_idx = int(basename.split("_")[1])

            # Load RGB
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))

            # print(osp.join(rgb_dir, basename + ".png"))
            # Load depthmap
            depthmap = cv2.imread(osp.join(depth_dir, basename.replace("frame_", "depth_") + ".tiff"), cv2.IMREAD_UNCHANGED)

            # Load dynamic_mask
            # print(osp.join(dymask_dir, basename.replace("frame_left_", "rigidmap_BW_left_") + ".png"))
            dynamic_mask_raw = self.get_dynamic_mask(
                seg=cv2.imread(osp.join(dymask_dir, basename.replace("frame_", "segmentation_") + ".png"), cv2.IMREAD_UNCHANGED),
                obj_positions=object_info["positions"][1:], # exclude background
                t=frame_idx,
                threshold=1e-3,
            )

            # load camera intrinsics and pose, TODO: load params if needed
            camera_pose = np.eye(4)
            intrinsics = np.array(
                [[35, 0.0, rgb_image.shape[1] / 2],
                    [0.0, 35, rgb_image.shape[0] / 2],
                    [0.0, 0.0, 1.0]]
            ) ## IT IS IN MILLIMETERS IN KUBRIC

            # Depth cropping if needed
            _, depthmap, _ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # Dynamic mask cropping if needed
            rgb_image, dynamic_mask, intrinsics = self._crop_resize_if_necessary(
                rgb_image, dynamic_mask_raw, intrinsics, resolution, rng=rng, info=view_idx
            )

            dynamic_mask = dynamic_mask.astype(bool)

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
                    dataset="kubric_dymask",
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

    def get_dynamic_mask(self, seg, obj_positions, t, threshold=1e-3):
        dynamic_mask = np.zeros(seg.shape, dtype=bool)

        for idx, pos in enumerate(obj_positions): # K-1, T, 3 0 is static background
            # if obj is moving
            if 0 <= t < pos.shape[0]:
                disp_sum = np.linalg.norm(pos[t] - pos[min(t+1, pos.shape[0]-1)])
                disp_sum += np.linalg.norm(pos[max(t-1, 0)] - pos[t])
            else:
                disp_sum = 0  # Default to zero if t is out of bounds
            if disp_sum > threshold:
                dynamic_mask |= (seg == idx+1)
        return dynamic_mask.astype(np.int8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_location",
        type=str,
        default='/mnt/rdata4_5/kubric_movi_f/',
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

    dataset = Kubric_DyMask(
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
