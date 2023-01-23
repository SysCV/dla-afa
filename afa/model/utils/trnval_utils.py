"""AFA training and validation utils.

# Code adapted from:
# https://github.com/NVIDIA/semantic-segmentation/blob/main/utils/trnval_utils.py # pylint: disable=line-too-long

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
from typing import List, Tuple

import torch

from afa.utils.structures import DictStrAny, NDArrayI64

from .seg_fix import seg_fix


def calc_err_mask_all(
    pred: NDArrayI64, gtruth: NDArrayI64, ignore_label: int
) -> NDArrayI64:
    """Calculate class-agnostic error masks."""
    mask = (gtruth >= 0) & (gtruth != ignore_label)
    err_mask = mask & (pred != gtruth)

    return err_mask.astype(int)  # type: ignore


def parse_output(
    dataset_name: str,
    ignore_label: int,
    image_names: List[str],
    output_dict: DictStrAny,
    output: torch.Tensor,
    gts: torch.Tensor,
    dataset_mode: str,
    dataset_root: str,
    do_seg_fix: bool = False,
    is_edge: bool = False,
) -> Tuple[NDArrayI64, DictStrAny]:
    """Parse mini batch output."""
    output_data = torch.nn.functional.softmax(output, dim=1).to("cpu").data
    edge_map = output_data[:, 1].numpy() if is_edge else None
    max_probs, predictions = output_data.max(1)

    # Assemble assets to visualize
    assets = {}
    for item in output_dict:
        if "attn_" in item:
            assets[item] = output_dict[item]
        if "pred_" in item:
            smax = torch.nn.functional.softmax(output_dict[item], dim=1)
            _, pred = smax.data.max(1)
            assets[item] = pred.to("cpu").numpy()

    predictions = predictions.numpy()

    # Seg-Fix for Cityscapes
    if dataset_name == "cityscapes" and do_seg_fix:
        predictions = seg_fix(
            predictions, image_names, dataset_mode, dataset_root
        )

    assets["prob_mask"] = max_probs
    if is_edge:
        assets["edge_map"] = edge_map

    if dataset_mode != "test":
        assets["err_mask"] = calc_err_mask_all(
            predictions,
            gts.to("cpu").numpy(),
            ignore_label=ignore_label,
        )

    return predictions, assets
