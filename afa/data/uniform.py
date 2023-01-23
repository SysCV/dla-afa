"""Class uniform for AFA.

# Code adapted from:
# https://github.com/NVIDIA/semantic-segmentation/blob/main/dataset/uniform.py

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

Uniform sampling of classes.
For all images, for all classes, generate centroids around which to sample.

All images are divided into tiles.
For each tile, a class can be present or not. If it is
present, calculate the centroid of the class and record it.

We would like to thank Peter Kontschieder for the inspiration of this idea.
"""
import json
import os
from collections import defaultdict
from functools import partial
from multiprocessing.dummy import Pool
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_info,
    rank_zero_only,
)
from scipy.ndimage.measurements import center_of_mass

from afa.utils.misc import Timer


def calc_tile_locations(
    tile_size: int, image_size: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Divide an image into tiles to help us cover classes that are spread out.

    tile_size: size of tile to distribute
    image_size: original image size
    return: locations of the tiles
    """
    image_size_y, image_size_x = image_size
    locations = []
    for y in range(image_size_y // tile_size):
        for x in range(image_size_x // tile_size):
            x_offs = x * tile_size
            y_offs = y * tile_size
            locations.append((x_offs, y_offs))
    return locations


def class_centroids_image(
    item: Tuple[str, str],
    tile_size: int,
    num_classes: int,
    id2trainid: Optional[Dict[int, int]],
    dropout_coarse_boost_classes: Optional[List[int]] = None,
) -> Dict[int, List[Tuple[str, str, List[int], int]]]:
    """For one image, calculate centroids for all classes present in image.

    item: image, image_name
    tile_size:
    num_classes:
    id2trainid: mapping from original id to training ids
    return: Centroids are calculated for each tile.
    """
    image_fn, label_fn = item
    centroids = defaultdict(list)
    mask = np.array(Image.open(label_fn))  # type: ignore
    image_size = mask.shape
    tile_locations = calc_tile_locations(tile_size, image_size)

    drop_mask = np.zeros((1024, 2048))
    drop_mask[15:840, 14:2030] = 1.0

    if "refinement" in label_fn:
        gt_coarse_mask_path = label_fn.replace(
            "refinement",
            os.path.join("gtCoarse", "gtCoarse"),
        )
        gt_coarse_mask_path = gt_coarse_mask_path.replace(
            "leftImg8bit", "gtCoarse_labelIds"
        )
        gt_coarse = np.array(Image.open(gt_coarse_mask_path))  # type: ignore

    mask_copy = mask.copy()
    if id2trainid is not None:
        for k, v in id2trainid.items():
            binary_mask = mask_copy == k
            if (
                "refinement" in label_fn
                and dropout_coarse_boost_classes is not None
                and v in dropout_coarse_boost_classes
                and binary_mask.sum() > 0
            ):
                binary_mask += gt_coarse == k
                binary_mask[binary_mask >= 1] = 1
                mask[binary_mask] = gt_coarse[binary_mask]
            mask[binary_mask] = v

    for x_offs, y_offs in tile_locations:
        patch = mask[y_offs : y_offs + tile_size, x_offs : x_offs + tile_size]
        for class_id in range(num_classes):
            if class_id in patch:
                patch_class = (patch == class_id).astype(int)
                centroid_y, centroid_x = center_of_mass(patch_class)
                centroid_y = int(centroid_y) + y_offs
                centroid_x = int(centroid_x) + x_offs
                centroids[class_id].append(
                    (image_fn, label_fn, [centroid_x, centroid_y], class_id)
                )
    return centroids


def class_centroids_all(
    items: List[Tuple[str, str]],
    num_classes: int,
    id2trainid: Optional[Dict[int, int]],
    tile_size: int,
    dropout_coarse_boost_classes: Optional[List[int]],
) -> Dict[int, List[Tuple[str, str, List[int], int]]]:
    """Calculate class centroids for all classes for all images for all tiles.

    items: list of (image_fn, label_fn)
    tile size: size of tile
    returns: dict that contains a list of centroids for each class
    """
    pool = Pool(os.cpu_count())
    class_centroids_item = partial(
        class_centroids_image,
        num_classes=num_classes,
        id2trainid=id2trainid,
        tile_size=tile_size,
        dropout_coarse_boost_classes=dropout_coarse_boost_classes,
    )

    centroids = defaultdict(list)
    new_centroids = pool.map(class_centroids_item, items)
    pool.close()
    pool.join()

    # combine each image's items into a single global dict
    for image_items in new_centroids:
        for class_id in image_items:
            centroids[class_id].extend(image_items[class_id])
    return centroids


@rank_zero_only
def _build_centroids(
    imgs: List[Tuple[str, str]],
    num_classes: int,
    id2trainid: Optional[Dict[int, int]],
    centroid_root: str,
    class_uniform_tile: int,
    json_fn: str,
    dropout_coarse_boost_classes: Optional[List[int]] = None,
) -> Dict[int, List[Tuple[str, str, List[int], int]]]:
    """Build centroids with rank zero only."""
    os.makedirs(centroid_root, exist_ok=True)

    # centroids is a dict (indexed by class) of lists of centroids
    centroids = class_centroids_all(
        imgs,
        num_classes,
        id2trainid=id2trainid,
        tile_size=class_uniform_tile,
        dropout_coarse_boost_classes=dropout_coarse_boost_classes,
    )
    with open(json_fn, "w", encoding="utf-8") as outfile:
        json.dump(centroids, outfile, indent=4)
    assert os.path.isfile(json_fn), f"Expected to find {json_fn}"

    return centroids


def build_centroids(
    dataset_name: str,
    imgs: List[Tuple[str, str]],
    num_classes: int,
    centroid_root: str,
    class_uniform_tile: int,
    dropout_coarse_boost_classes: Optional[List[int]] = None,
    cv: Optional[int] = None,
    coarse: bool = False,
    id2trainid: Optional[Dict[int, int]] = None,
) -> Dict[int, List[Tuple[str, str, List[int], int]]]:
    """The first step of uniform sampling is to decide sampling centers.

    The idea is to divide each image into tiles and within each tile,
    we compute a centroid for each class to indicate roughly where to
    sample a crop during training.

    This function computes these centroids and returns a list of them.
    """
    centroid_fn = dataset_name

    if coarse:
        centroid_fn += "_coarse"
    elif cv is not None:
        centroid_fn += f"_cv{cv}"

    centroid_fn += f"_tile{class_uniform_tile}.json"
    json_fn = os.path.join(centroid_root, centroid_fn)

    t = Timer()
    if os.path.isfile(json_fn):
        rank_zero_info(f"Loading centroid file {json_fn}...")
        with open(json_fn, "r", encoding="utf-8") as json_data:
            centroids = json.load(json_data)
        centroids = {int(idx): centroids[idx] for idx in centroids}
        rank_zero_info(f"Found {len(centroids)} centroids")
    else:
        rank_zero_info(
            f"Do not find centroid file {json_fn}, so building it..."
        )
        centroids = _build_centroids(
            imgs=imgs,
            num_classes=num_classes,
            id2trainid=id2trainid,
            centroid_root=centroid_root,
            class_uniform_tile=class_uniform_tile,
            json_fn=json_fn,
            dropout_coarse_boost_classes=dropout_coarse_boost_classes,
        )

    rank_zero_info(
        f"Load centroid file {json_fn} takes {t.time():.2f} seconds."
    )
    return centroids  # type: ignore


def random_sampling(
    alist: Union[List[Tuple[str, str]], List[Tuple[str, str, List[int], int]]],
    num: int,
) -> List[Union[Tuple[str, str], Tuple[str, str, List[int], int]]]:
    """Randomly sample num items from the list.

    alist: list of centroids to sample from
    num: can be larger than the list and if so, then wrap around
    return: class uniform samples from the list
    """
    sampling = []
    len_list = len(alist)
    assert len_list, "len_list is zero!"
    indices = np.arange(len_list)
    np.random.shuffle(indices)

    for i in range(num):
        item = alist[indices[i % len_list]]
        sampling.append(item)
    return sampling


def build_epoch(
    imgs: List[Tuple[str, str]],
    centroids: Dict[
        int,
        List[Tuple[str, str, List[int], int]],
    ],
    num_classes: int,
    class_uniform_pct: float,
) -> List[Union[Tuple[str, str], Tuple[str, str, List[int], int]]]:
    """Generate an epoch of crops using uniform sampling.

    Needs to be called every epoch.
    Will not apply uniform sampling if not train or class uniform is off.

    Inputs:
      imgs - list of imgs
      centroids - list of class centroids
      num_classes - number of classes
      class_uniform_pct: % of uniform images in one epoch
    Outputs:
      imgs - list of images to use this epoch
    """
    rank_zero_info(f"Class Uniform Percentage: {str(class_uniform_pct)}")
    num_epoch = int(len(imgs))

    rank_zero_info(f"Class Uniform items per Epoch: {str(num_epoch)}")
    num_per_class = int((num_epoch * class_uniform_pct) / num_classes)
    class_uniform_count = num_per_class * num_classes
    num_rand = num_epoch - class_uniform_count
    # create random crops
    imgs_uniform = random_sampling(imgs, num_rand)

    # now add uniform sampling
    for class_id in range(num_classes):
        rank_zero_info(f"cls {class_id} len {len(centroids[class_id])}")
    for class_id in range(num_classes):
        num_per_class_biased = num_per_class
        centroid_len = len(centroids[class_id])
        if centroid_len == 0:
            pass
        else:
            class_centroids = random_sampling(
                centroids[class_id], num_per_class_biased
            )
            imgs_uniform.extend(class_centroids)

    return imgs_uniform
