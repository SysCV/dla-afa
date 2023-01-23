"""OCR block."""
from typing import Tuple

import torch
from torch import nn

from ..utils.mynn import bn_relu, initialize_weights
from ..utils.ocr_utils import SpatialGatherModule, SpatialOCRModule


class OCRBlock(nn.Module):  # type: ignore
    """OCR Block.

    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """

    def __init__(
        self,
        num_classes: int,
        high_level_ch: int,
        ocr_mid_channels: int,
        ocr_key_channels: int,
    ) -> None:
        """Init."""
        super().__init__()

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(
                high_level_ch,
                ocr_mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            bn_relu(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGatherModule(num_classes)
        self.ocr_distri_head = SpatialOCRModule(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.05,
        )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(
                high_level_ch,
                high_level_ch,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            bn_relu(high_level_ch),
            nn.Conv2d(
                high_level_ch,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        initialize_weights(
            self.conv3x3_ocr,
            self.ocr_gather_head,
            self.ocr_distri_head,
            self.cls_head,
            self.aux_head,
        )

    def forward(
        self, high_level_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward."""
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats
