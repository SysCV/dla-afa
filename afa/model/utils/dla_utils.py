"""Utlis for DLA."""
import math
from typing import Generator, List, Tuple

import torch
from inplace_abn import ABN
from torch import nn

from .ops import resize_tensor


class Identity(nn.Module):  # type: ignore
    """Identity module."""

    @staticmethod
    def _fwd(feat: torch.Tensor) -> torch.Tensor:
        """Simple forward."""
        return feat

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self._fwd(feat)


def fill_up_weights(up: nn.Module) -> None:
    """Fill up model weights."""
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (
                1 - math.fabs(j / f - c)
            )
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def dla_seg_head(
    first_level: int, num_classes: int, channels: List[int]
) -> Tuple[nn.Sequential, nn.Module, nn.Module]:
    """Built DLA segmentation head."""
    fc = nn.Sequential(
        nn.Conv2d(
            channels[first_level],
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
    )

    up_factor = 2**first_level
    if up_factor > 1:
        up = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            up_factor * 2,
            stride=up_factor,
            padding=up_factor // 2,
            output_padding=0,
            groups=num_classes,
            bias=False,
        )
        fill_up_weights(up)
        up.weight.requires_grad = False
    else:
        up = Identity()

    return fc, up, nn.LogSoftmax(dim=1)


def init_weights(modules: Generator[nn.Module, None, None]) -> None:
    """Initialize module weights."""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, (ABN, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def dla_resize(feat: torch.Tensor, training: bool = False) -> torch.Tensor:
    """Resize for DLA model."""
    img_h, img_w = feat.shape[2:]

    if training:
        base = 64
    else:
        base = 128

    if img_w % base == 0:
        w = img_w
    else:
        w = (img_w // base + 1) * base

    if img_h % base == 0:
        h = img_h
    else:
        h = (img_h // base + 1) * base

    return resize_tensor(
        feat,
        target_size=(h, w),
    )
