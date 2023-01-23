"""Generic edge dataloader base class for AFA."""
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from afa.utils.structures import ArgsType, NDArrayF32

from .base import BaseDataset


class BaseEdgeDataset(BaseDataset):
    """Base boundary detection dataset."""

    def __init__(
        self,
        *args: ArgsType,
        **kwargs: ArgsType,
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.trainid_to_name: Dict[int, str] = {}
        self.images: List[str] = []
        self.masks: List[str] = []

    def do_transforms(  # type:ignore
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Do transformations to image and mask."""
        scale_float = 1.0

        if self.joint_transform_list is not None:
            for xform in self.joint_transform_list:
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

    @staticmethod
    def _get_boundaries_target(filepath: str) -> Image.Image:
        """Get boundaries target."""
        target: NDArrayF32 = np.array(Image.open(filepath), dtype=np.float32)
        if len(target.shape) == 2:
            target = target / 255.0
        else:
            target = target[:, :, 0] / 255.0
        target = Image.fromarray(target.astype(np.uint8))
        return target

    def __getitem__(self, index: int):  # type:ignore
        """Generate data."""
        img = Image.open(self.images[index]).convert("RGB")

        if self.mode == "train":
            target = self._get_boundaries_target(self.masks[index])
        else:
            w, h = img.size
            target = Image.fromarray(np.zeros((h, w)))

        img, mask, scale_float = self.do_transforms(img, target)

        img_name = self.images[index].split("/")[-1]
        return img, mask, img_name, scale_float

    def __len__(self) -> int:
        """Length."""
        return len(self.images)
