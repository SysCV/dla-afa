"""Generic dataloader base class for AFA.

# Code adapted from:
# https://github.com/NVIDIA/semantic-segmentation/blob/main/datasets/base_loader.py # pylint: disable=line-too-long

Source License
# Copyright 2020 Nvidia Corporation

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this # pylint: disable=line-too-long
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
import glob
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils import data

from afa.utils.structures import ArgsType, AugmentType, NDArrayI64

from ..uniform import build_epoch


class BaseDataset(data.Dataset):  # type: ignore
    """Base semantic segmenation dataset."""

    def __init__(  # pylint: disable=unused-argument
        self,
        *args: ArgsType,
        name: str,
        asset_dir: str,
        mode: str,
        num_classes: int,
        ignore_label: int,
        mean: List[float],
        std: List[float],
        class_uniform_pct: float = 0.5,
        class_uniform_tile: int = 1024,
        joint_transform_list: Optional[List[AugmentType]] = None,
        img_transform: Optional[AugmentType] = None,
        label_transform: Optional[AugmentType] = None,
        folder: Optional[str] = None,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__()
        self.name = name
        self.asset_dir = asset_dir
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.joint_transform_list = joint_transform_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.folder = folder

        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.centroid_root = os.path.join(asset_dir, "uniform_centroids")

        self.centroids: Dict[
            int,
            List[Tuple[str, str, List[int], int]],
        ]
        self.train = mode == "train"
        self.id_to_trainid: Dict[int, int] = {}
        self.imgs: List[
            Union[Tuple[str, str], Tuple[str, str, List[int], int]]
        ]
        self.all_imgs: List[Tuple[str, str]]
        self.color_mapping: List[int]

        self.mask_out_cityscapes = False
        self.custom_coarse_prob: Optional[float] = None
        self.dropout_coarse_boost_classes: Optional[List[int]] = None
        self.drop_mask = np.zeros((1024, 2048))
        self.drop_mask[15:840, 14:2030] = 1.0

    def build_epoch(self) -> None:
        """For class uniform sampling.

        Every epoch, we want to recompute which tiles from which images we want
        to sample from, so that the sampling is uniformly random.
        """
        if self.class_uniform_pct and self.train:
            self.imgs = build_epoch(
                imgs=self.all_imgs,
                centroids=self.centroids,
                num_classes=self.num_classes,
                class_uniform_pct=self.class_uniform_pct,
            )

    def find_images(
        self,
        img_root: str,
        mask_root: str,
        img_ext: str,
        mask_ext: str,
    ) -> List[Tuple[str, str]]:
        """Find image and segmentation mask files."""
        img_path = f"{img_root}/*.{img_ext}"
        imgs = glob.glob(img_path)
        items = []
        for full_img_fn in imgs:
            _, img_fn = os.path.split(full_img_fn)
            img_name, _ = os.path.splitext(img_fn)
            if self.mode == "test":
                full_mask_fn = ""
            else:
                full_mask_fn = f"{img_name}.{mask_ext}"
                full_mask_fn = os.path.join(mask_root, full_mask_fn)
                if not os.path.exists(full_mask_fn):
                    continue
            items.append((full_img_fn, full_mask_fn))
        return items

    def colorize_mask(self, image_array: NDArrayI64) -> Image.Image:
        """Colorize the mask."""
        new_mask = Image.fromarray(image_array.astype(np.uint8)).convert("P")
        new_mask.putpalette(self.color_mapping)
        return new_mask

    def do_transforms(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        centroid: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Do transformations to image and mask."""
        scale_float = 1.0

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    outputs = xform(img, mask, centroid)  # type: ignore
                else:
                    outputs = xform(img, mask)  # type: ignore

                if len(outputs) == 3:
                    img, mask, scale_float = outputs
                else:
                    img, mask = outputs

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)

        return img, mask, scale_float

    def read_images(
        self, img_path: str, mask_path: str, mask_out: bool = False
    ) -> Tuple[Image.Image, Image.Image, str]:
        """Read images."""
        img = Image.open(img_path).convert("RGB")
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if mask_path == "":
            w, h = img.size
            mask = np.zeros((h, w))
        else:
            mask = Image.open(mask_path)

            # This code is specific to cityscapes
            if "refinement" in mask_path:
                gt_coarse_mask_path = mask_path.replace(
                    "refinement",
                    os.path.join("gtCoarse", "gtCoarse"),
                )
                gt_coarse_mask_path = gt_coarse_mask_path.replace(
                    "leftImg8bit", "gtCoarse_labelIds"
                )
                gt_coarse = np.array(Image.open(gt_coarse_mask_path))  # type: ignore # pylint: disable=line-too-long

            mask = np.array(mask)
            if mask_out:
                mask = self.drop_mask * mask

            mask = mask.copy()
            for k, v in self.id_to_trainid.items():
                binary_mask = mask == k
                if (
                    "refinement" in mask_path
                    and self.dropout_coarse_boost_classes is not None
                    and v in self.dropout_coarse_boost_classes
                    and binary_mask.sum() > 0
                    and "vidseq" not in mask_path
                ):
                    binary_mask += gt_coarse == k
                    binary_mask[binary_mask >= 1] = 1
                    mask[binary_mask] = gt_coarse[binary_mask]
                mask[binary_mask] = v

            mask = Image.fromarray(mask.astype(np.uint8))
        return img, mask, img_name

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, float]:
        """Generate data.

        :return:
        - image: image, tensor
        - mask: mask, tensor
        - image_name: basename of file, string
        """
        # Pick an image, fill in defaults if not using class uniform
        if len(self.imgs[index]) == 2:
            img_path, mask_path = self.imgs[index]  # type: ignore
            centroid = None
        else:
            img_path, mask_path, centroid, _ = self.imgs[index]  # type: ignore

        mask_out = (
            self.mask_out_cityscapes
            and self.custom_coarse_prob is not None
            and "refinement" in mask_path
        )

        img, mask, img_name = self.read_images(
            img_path, mask_path, mask_out=mask_out
        )

        ######################################################################
        # Thresholding is done when using coarse-labelled Cityscapes images
        ######################################################################
        if mask_path is not None and "refinement" in mask_path:
            mask = np.array(mask)
            prob_mask_path = mask_path.replace(".png", "_prob.png")
            # put it in 0 to 1
            prob_map = np.array(Image.open(prob_mask_path)) / 255.0
            prob_map_threshold = prob_map < self.custom_coarse_prob
            mask[prob_map_threshold] = self.ignore_label
            mask = Image.fromarray(mask.astype(np.uint8))

        img, mask, scale_float = self.do_transforms(img, mask, centroid)

        return img, mask, img_name, scale_float

    def __len__(self) -> int:
        """Length."""
        return len(self.all_imgs)
