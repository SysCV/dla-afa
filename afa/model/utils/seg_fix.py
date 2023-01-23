"""SegFix for cityscapes.

# Code adapted from:
# https://github.com/openseg-group/openseg.pytorch/blob/master/scripts/cityscapes/segfix.py # pylint: disable=line-too-long
"""
import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import io

from afa.utils.structures import NDArrayF32, NDArrayI64


def gen_coord_map(height: int, width: int) -> Tuple[int, int]:
    """Generate coord map."""
    coord_vecs = [
        torch.arange(length, dtype=torch.float) for length in (height, width)
    ]
    coord_h, coord_w = torch.meshgrid(coord_vecs, indexing="ij")
    return coord_h, coord_w


def shift(input_x: NDArrayI64, offset: NDArrayF32) -> NDArrayI64:
    """Shift input with offset.

    x: h x w
    offset: 2 x h x w
    """
    h, w = input_x.shape
    x = torch.from_numpy(input_x).unsqueeze(0)
    offset = torch.from_numpy(offset).unsqueeze(0)
    coord_map = gen_coord_map(h, w)
    norm_factor = torch.FloatTensor([(w - 1) / 2, (h - 1) / 2])
    grid_h = offset[:, 0] + coord_map[0]
    grid_w = offset[:, 1] + coord_map[1]
    grid = torch.stack([grid_w, grid_h], dim=-1) / norm_factor - 1
    x = (
        F.grid_sample(
            x.unsqueeze(1).float(),
            grid,
            padding_mode="border",
            mode="bilinear",
            align_corners=True,
        )
        .squeeze()
        .numpy()
    )
    x = np.round(x)
    return x.astype(np.int64)  # type: ignore


def get_offset(offset_dir: str, basename: str, scale: int = 2) -> NDArrayF32:
    """Get offset."""
    return (  # type: ignore
        io.loadmat(osp.join(offset_dir, f"{basename}.mat"))["mat"]
        .astype(np.float32)
        .transpose(2, 0, 1)
        * scale
    )


def seg_fix(
    predictions: NDArrayI64,
    img_names: List[str],
    split: str,
    cityscapes_root: str,
) -> NDArrayI64:
    """Seg fix post-processing."""
    seg_fix_root = osp.join(cityscapes_root, "offset_semantic")

    if split == "val":
        offset_dir = osp.join(
            seg_fix_root,
            "val",
            "offset_pred",
            "semantic",
            "offset_hrnext",
        )
    else:
        offset_dir = osp.join(
            seg_fix_root,
            "test_offset",
            "semantic",
            "offset_hrnext",
        )

    preds = []
    for i in range(predictions.shape[0]):
        input_label_map = predictions[i]
        basename = img_names[i]

        offset_map = get_offset(offset_dir, basename)
        output_label_map = shift(input_label_map, offset_map)

        preds.append(output_label_map)

    return np.asarray(preds)
