"""Mapillary Vistas dataset.

# Code adapted from:
# https://github.com/NVIDIA/semantic-segmentation/blob/main/dataset/mapillary.py # pylint: disable=line-too-long

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
import json
from os import path
from typing import List

from pytorch_lightning.utilities.rank_zero import rank_zero_info

from afa.utils.structures import ArgsType

from ..uniform import build_centroids
from .base import BaseDataset
from .custom import make_dataset_folder


class MapillaryVistas(BaseDataset):
    """Mapillary Vistas dataset class."""

    def __init__(
        self,
        *args: ArgsType,
        **kwargs: ArgsType,
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.root = path.join(self.asset_dir, "data/Mapillary")
        config_fn = path.join(self.root, "config.json")
        self.fill_colormap_and_names(config_fn)

        ######################################################################
        # Assemble image lists
        ######################################################################
        if self.folder is not None:
            self.all_imgs = make_dataset_folder(self.folder)
        else:
            mode_to_splits = {
                "train": "training",
                "val": "validation",
                "test": "testing",
            }
            split_name = mode_to_splits[self.mode]
            img_root = path.join(self.root, split_name, "images")
            mask_root = path.join(self.root, split_name, "v1.2/labels")
            self.all_imgs = self.find_images(
                img_root,
                mask_root,
                img_ext="jpg",
                mask_ext="png",
            )

            rank_zero_info(
                f"mode {self.mode} found {len(self.all_imgs)} images"
            )

        if self.mode == "train" and self.class_uniform_pct:
            self.centroids = build_centroids(
                dataset_name=self.name,
                imgs=self.all_imgs,
                num_classes=self.num_classes,
                centroid_root=self.centroid_root,
                class_uniform_tile=self.class_uniform_tile,
            )

        self.imgs = self.all_imgs  # type: ignore

    def fill_colormap_and_names(self, config_fn: str) -> None:
        """Mapillary code for color map and class names."""
        with open(config_fn, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        config_labels = config["labels"]

        # calculate label color mapping
        colormap: List[int] = []
        self.trainid_to_name = {}
        for i, label in enumerate(config_labels):
            colormap = colormap + label["color"]
            name = label["readable"]
            name = name.replace(" ", "_")
            self.trainid_to_name[i] = name
        self.color_mapping = colormap
