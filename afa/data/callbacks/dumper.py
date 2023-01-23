"""Image dumper for AFA.

# Code adapted from:
# https://github.com/NVIDIA/semantic-segmentation/blob/main/utils/misc.py

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
import os
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
from pytorch_lightning.callbacks import Callback
from scipy.io import savemat

from afa.data.datasets import BaseDataset, cityscapes_label2trainid
from afa.utils.structures import ArgsType, DictStrAny, NDArrayF32, NDArrayI64


class ImageDumperCallback(Callback):
    """Image dumping callback.

    Pass images / tensors from pipeline and it first converts them to images
    (doing transformations where necessary) and then writes out to the disk.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        output_dir: str,
        dump_for_submission: bool = False,
        dump_assets: bool = False,
        dump_gts: bool = True,
        is_edge: bool = False,
    ) -> None:
        """Init."""
        self.dataset = dataset
        self.dump_for_submission = dump_for_submission
        self.dump_assets = dump_assets
        self.is_edge = is_edge
        self.dump_gts = dump_gts

        if dump_for_submission:
            self.output_dir = os.path.join(output_dir, "submit")
        else:
            self.output_dir = os.path.join(output_dir, "best_images")
        os.makedirs(self.output_dir, exist_ok=True)

        inv_mean = [-m / s for m, s in zip(dataset.mean, dataset.std)]
        inv_std = [1 / s for s in dataset.std]
        self.inv_normalize = standard_transforms.Normalize(
            mean=inv_mean, std=inv_std
        )

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ArgsType,
        batch: ArgsType,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Hook for on_predict_batch_end."""
        for idx in range(len(outputs["img_names"])):
            img_name = outputs["img_names"][idx]
            input_image = outputs["input_images"][idx]
            gt_image = outputs["gt_images"][idx]
            prediction = outputs["predictions"][idx]
            if self.is_edge:
                self.write_edge(img_name, outputs["assets"]["edge_map"][idx])
            elif self.dump_for_submission:
                self.dump_submission(img_name, prediction)
            else:
                self.dump(img_name, input_image, gt_image, prediction)

                if self.dump_assets:
                    self.dump_asset_images(img_name, outputs["assets"], idx)

    def dump(
        self,
        img_name: str,
        input_image: Image.Image,
        gt_image: torch.Tensor,
        prediction: NDArrayI64,
    ) -> None:
        """Dump images."""
        input_image = self.inv_normalize(input_image)
        input_image = input_image.cpu()
        input_image = standard_transforms.ToPILImage()(input_image)
        input_image = input_image.convert("RGB")
        input_image_fn = f"{img_name}_input.png"
        input_image.save(os.path.join(self.output_dir, input_image_fn))

        if self.dump_gts:
            gt_fn = f"{img_name}_gt.png"
            gt_pil = self.dataset.colorize_mask(gt_image.cpu().numpy())
            gt_pil.save(os.path.join(self.output_dir, gt_fn))

        prediction_fn = f"{img_name}_prediction.png"
        prediction_pil = self.dataset.colorize_mask(prediction)
        prediction_pil.save(os.path.join(self.output_dir, prediction_fn))

        prediction_pil = prediction_pil.convert("RGB")
        composited = Image.blend(input_image, prediction_pil, 0.4)
        composited_fn = f"composited_{img_name}.png"
        composited_fn = os.path.join(self.output_dir, composited_fn)
        composited.save(composited_fn)

    def dump_submission(self, img_name: str, prediction: NDArrayI64) -> None:
        """Dump for submission."""
        submit_fn = f"{img_name}.png"
        if self.dataset.name == "cityscapes":
            label_out = np.zeros_like(prediction)
            for label_id, train_id in cityscapes_label2trainid.items():
                label_out[np.where(prediction == train_id)] = label_id
            label_out = label_out.astype(np.uint8)
        else:
            label_out = prediction.astype(np.uint8)
        Image.fromarray(label_out).save(
            os.path.join(self.output_dir, submit_fn)
        )

    def dump_asset_images(
        self, img_name: str, assets: DictStrAny, idx: int
    ) -> None:
        """Dump assets."""
        for asset in assets:
            mask = assets[asset][idx]
            mask_fn = os.path.join(self.output_dir, f"{img_name}_{asset}.png")

            if "pred_" in asset:
                pred_pil = self.dataset.colorize_mask(mask)
                pred_pil.save(mask_fn)
            else:
                draw_attn(mask, mask_fn)

    def write_edge(
        self,
        img_name: str,
        edge_map: torch.Tensor,
    ) -> None:
        """Write boundary detection results."""
        if ".jpg" in img_name:
            # BSDS500
            img_name = img_name.split(".jpg")[0]
        else:
            # NYUD
            img_name = img_name.split(".png")[0]
        edge_path = f"{img_name}.png"
        edge_map_img = (edge_map * 255.0).astype(np.uint8)
        edge_map_img = standard_transforms.ToPILImage()(edge_map_img)
        edge_map_img = edge_map_img.convert("RGB")
        os.makedirs(os.path.join(self.output_dir, "edges"), exist_ok=True)
        edge_map_img.save(os.path.join(self.output_dir, "edges", edge_path))
        os.makedirs(os.path.join(self.output_dir, "mats"), exist_ok=True)
        savemat(
            os.path.join(self.output_dir, "mats", edge_path).replace(
                ".png", ".mat"
            ),
            {"img": edge_map},
        )


def draw_attn(mask: Union[torch.Tensor, NDArrayF32], file_path: str) -> None:
    """Visualize the attention map."""
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    else:
        mask = mask.squeeze()
    mask = mask * 255
    mask = mask.astype(np.uint8)
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.convert("RGB")
    mask_pil.save(file_path)
