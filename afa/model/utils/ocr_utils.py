"""OCR utils.

# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
#
# This code is from: https://github.com/HRNet/HRNet-Semantic-Segmentation
# -----------------------------------------------------------------------------
"""
import torch
import torch.nn.functional as F
from torch import nn

from .mynn import bn_relu


class SpatialGatherModule(nn.Module):  # type: ignore
    """Spatial gather module.

    Aggregate the context features according to the initial
    predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.

    Output:
      The correlation of every class map with every feature map
      shape = [n, num_feats, num_classes, 1]
    """

    def __init__(self, cls_num: int, scale: int = 1) -> None:
        """Init."""
        super().__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(
        self, feats: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """Forward."""
        batch_size, c, _, _ = (
            probs.size(0),
            probs.size(1),
            probs.size(2),
            probs.size(3),
        )

        # each class image now a vector
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)

        # batch x hw x c
        feats = feats.permute(0, 2, 1)

        # batch x k x hw
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(nn.Module):  # type: ignore
    """The basic implementation for object context block.

    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature
                            maps (save memory cost)

    Return:
        N X C X H X W
    """

    def __init__(
        self, in_channels: int, key_channels: int, scale: int = 1
    ) -> None:
        """Init."""
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            bn_relu(self.key_channels),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            bn_relu(self.key_channels),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            bn_relu(self.key_channels),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            bn_relu(self.key_channels),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            bn_relu(self.key_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            bn_relu(self.in_channels),
        )

    def forward(self, feat: torch.Tensor, proxy: torch.Tensor) -> torch.Tensor:
        """Forward."""
        batch_size, h, w = feat.size(0), feat.size(2), feat.size(3)
        if self.scale > 1:
            feat = self.pool(feat)

        query = self.f_pixel(feat).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *feat.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(
                input=context,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        return context


class SpatialOCRModule(nn.Module):  # type: ignore
    """Implementation of the OCR module.

    We aggregate the global object representation to update the representation
    for each pixel.
    """

    def __init__(
        self,
        in_channels: int,
        key_channels: int,
        out_channels: int,
        scale: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Init."""
        super().__init__()
        self.object_context_block = ObjectAttentionBlock(
            in_channels, key_channels, scale
        )
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(
                _in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            bn_relu(out_channels),
            nn.Dropout2d(dropout),
        )

    def forward(
        self, feats: torch.Tensor, proxy_feats: torch.Tensor
    ) -> torch.Tensor:
        """Forward."""
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output
