"""Cityscapes dataset.

# Code adapted from:
# https://github.com/NVIDIA/semantic-segmentation/blob/main/dataset/cityscapes.py # pylint: disable=line-too-long

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
import copy
import itertools
import os
from typing import List, Tuple

from pytorch_lightning.utilities.rank_zero import rank_zero_info

from afa.utils.structures import ArgsType

from ..uniform import build_centroids
from .base import BaseDataset
from .cityscapes_labels import label2trainid, trainId2name
from .custom import make_dataset_folder

coarse_boost_classes = [
    3,
    4,
    6,
    7,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
]
dropout_coarse_boost_classes = [14, 15, 16]


class CityScapes(BaseDataset):
    """CityScapes dataset class."""

    def __init__(
        self,
        *args: ArgsType,
        cv: int = 0,
        cityscapes_splits: int = 3,
        only_fine: bool = False,
        custom_coarse_prob: float = 0.5,
        mask_out_cityscapes: bool = True,
        **kwargs: ArgsType,
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.root = os.path.join(self.asset_dir, "data/Cityscapes")
        self.cv = cv
        self.only_fine = only_fine
        self.cityscapes_splits = cityscapes_splits
        self.custom_coarse_prob = custom_coarse_prob
        self.dropout_coarse_boost_classes = dropout_coarse_boost_classes
        self.mask_out_cityscapes = mask_out_cityscapes

        self.id_to_trainid = label2trainid
        self.trainid_to_name = trainId2name
        self.fill_colormap()

        # Fine
        img_root = os.path.join(
            self.root, "leftImg8bit_trainvaltest/leftImg8bit"
        )
        mask_root = os.path.join(self.root, "gtFine_trainvaltest/gtFine")
        if self.folder is not None:
            self.all_imgs = make_dataset_folder(self.folder)
        elif self.mode == "test":
            self.test_cities = self.cities_cv_split()
            self.all_imgs = self.find_cityscapes_images(
                cities=self.test_cities,
                img_root=img_root,
                mask_root=mask_root,
                img_ext="png",
            )
        else:
            self.fine_cities = self.cities_cv_split()
            self.all_imgs = self.find_cityscapes_images(
                cities=self.fine_cities,
                img_root=img_root,
                mask_root=mask_root,
                img_ext="png",
            )

        if self.mode == "train" and self.class_uniform_pct:
            self.fine_centroids = build_centroids(
                dataset_name=self.name,
                imgs=self.all_imgs,
                num_classes=self.num_classes,
                centroid_root=self.centroid_root,
                class_uniform_tile=self.class_uniform_tile,
                cv=cv,
                id2trainid=self.id_to_trainid,
            )
            self.centroids = copy.deepcopy(self.fine_centroids)

            # Coarse
            if not only_fine:
                self.cityscapes_customcoarse = os.path.join(
                    self.root, "refinement"
                )
                self.coarse_cities = self.find_coarse_cities()
                img_root = os.path.join(
                    self.root, "leftImg8bit_trainextra/leftImg8bit"
                )
                mask_root = os.path.join(self.root, "gtCoarse", "gtCoarse")
                self.coarse_imgs = self.find_cityscapes_images(
                    cities=self.coarse_cities,
                    img_root=img_root,
                    mask_root=mask_root,
                    img_ext="png",
                    fine_coarse="gtCoarse",
                )

                self.coarse_centroids = build_centroids(
                    dataset_name=self.name,
                    imgs=self.coarse_imgs,
                    num_classes=self.num_classes,
                    centroid_root=self.centroid_root,
                    class_uniform_tile=self.class_uniform_tile,
                    dropout_coarse_boost_classes=self.dropout_coarse_boost_classes,  # pylint: disable=line-too-long
                    coarse=True,
                    id2trainid=self.id_to_trainid,
                )
                for cid in coarse_boost_classes:
                    self.centroids[cid].extend(self.coarse_centroids[cid])

        self.imgs = self.all_imgs  # type: ignore

    def disable_coarse(self) -> None:
        """Turn off using coarse images in training."""
        self.centroids = copy.deepcopy(self.fine_centroids)

    def find_cityscapes_images(
        self,
        cities: List[str],
        img_root: str,
        mask_root: str,
        img_ext: str,
        fine_coarse: str = "gtFine",
    ) -> List[Tuple[str, str]]:
        """Find image and segmentation mask.

        Inputs:
        img_root: path to parent directory of train/val/test dirs
        mask_root: path to parent directory of train/val/test dirs
        img_ext: image file extension
        mask_ext: mask file extension
        cities: a list of cities, each element in the form of 'train/a_city'
          or 'val/a_city', for example.
        """
        items = []
        for city in cities:
            img_dir = f"{img_root}/{city}"
            for file_name in os.listdir(img_dir):
                basename, ext = os.path.splitext(file_name)
                assert ext == "." + img_ext, f"{ext} {img_ext}"
                full_img_fn = os.path.join(img_dir, file_name)
                basename, ext = file_name.split("_leftImg8bit")
                if self.custom_coarse_prob and fine_coarse != "gtFine":
                    mask_fn = f"{basename}_leftImg8bit.png"
                    full_mask_fn = os.path.join(
                        self.cityscapes_customcoarse, city, mask_fn
                    )
                    os.path.isfile(full_mask_fn)
                else:
                    mask_fn = f"{basename}_{fine_coarse}_labelIds{ext}"
                    full_mask_fn = os.path.join(mask_root, city, mask_fn)
                items.append((full_img_fn, full_mask_fn))

        rank_zero_info(f"mode {self.mode} found {len(items)} images")

        return items

    def fill_colormap(self) -> None:
        """Build colormap."""
        palette_array = [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]
        palette = list(itertools.chain.from_iterable(palette_array))
        zero_pad = 256 * 3 - len(palette)
        for _ in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette

    def cities_cv_split(self) -> List[str]:
        """Find cities that correspond to a given split of the data.

        We split the data such that a given city belongs to either train or val
        but never both. cv0 is defined to be the default split.

        all_cities = [x x x x x x x x x x x x]
        val:
            split0     [x x x                  ]
            split1     [        x x x          ]
            split2     [                x x x  ]
        trn:
            split0     [      x x x x x x x x x]
            split1     [x x x x       x x x x x]
            split2     [x x x x x x x x        ]

        cv split: 0, 1, 2, 3
        cv split == 3 means use train + val
        """
        if self.mode == "test":
            test_path = os.path.join(
                self.root, "leftImg8bit_trainvaltest/leftImg8bit", "test"
            )
            test_cities = ["test/" + c for c in os.listdir(test_path)]
            all_cities = test_cities
            rank_zero_info(f"Submission {all_cities}")
            cities = all_cities
        else:
            if self.mode == "val":
                self.cv = 0

            trn_path = os.path.join(
                self.root, "leftImg8bit_trainvaltest/leftImg8bit", "train"
            )
            val_path = os.path.join(
                self.root, "leftImg8bit_trainvaltest/leftImg8bit", "val"
            )

            trn_cities = ["train/" + c for c in os.listdir(trn_path)]
            # sort to insure reproducibility
            trn_cities = sorted(trn_cities)
            val_cities = ["val/" + c for c in os.listdir(val_path)]

            all_cities = val_cities + trn_cities

            if self.cv == 3:
                cities = all_cities
            else:
                num_val_cities = len(val_cities)
                num_cities = len(all_cities)

                offset = self.cv * num_cities // self.cityscapes_splits
                cities = []
                for j in range(num_cities):
                    if offset <= j < (offset + num_val_cities):
                        if self.mode == "val":
                            cities.append(all_cities[j])
                    else:
                        if self.mode == "train":
                            cities.append(all_cities[j])
            rank_zero_info(f"cv split {self.cv} for {self.mode} {cities}")
        return cities

    def find_coarse_cities(self) -> List[str]:
        """Find coarse cities."""
        coarse_path = os.path.join(
            self.root, "leftImg8bit_trainextra/leftImg8bit", "train_extra"
        )
        found_coarse_cities = [
            "train_extra/" + c for c in os.listdir(coarse_path)
        ]

        rank_zero_info(f"Found {len(found_coarse_cities)} coarse cities")
        return found_coarse_cities
