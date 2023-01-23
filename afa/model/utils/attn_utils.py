"""Attention utils."""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .mynn import bn_relu, initialize_weights


class SpatialAttentionModule(nn.Module):  # type: ignore
    """Spatial Attention module."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bot_channels: int,
        activation: str = "sigmoid",
        inplace: bool = True,
        kernel_size: int = 3,
    ) -> None:
        """Init."""
        super().__init__()

        self.activation = activation

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                bot_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            bn_relu(bot_channels, inplace=inplace),
        )

        self.conv_inner = nn.Sequential(
            nn.Conv2d(
                bot_channels,
                bot_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            bn_relu(bot_channels, inplace=inplace),
        )

        self.final = nn.Sequential(
            nn.Conv2d(bot_channels, out_ch, kernel_size=1, bias=False),
        )

        if self.activation == "relu":
            self.activation_layer = nn.Sequential(nn.ReLU())
        elif self.activation == "sigmoid":
            self.activation_layer = nn.Sequential(nn.Sequential(nn.Sigmoid()))

        initialize_weights(self.conv, self.conv_inner, self.final)

    def forward(
        self, high_level_features: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Forward."""
        feats = self.conv(high_level_features)
        feats = self.conv_inner(feats)
        attn_logits = self.final(feats)

        attn = None
        if self.activation == "relu":
            attn_logits = self.activation_layer(attn_logits)
        elif self.activation == "abs":
            attn_logits = torch.abs(attn_logits)
        else:
            attn = self.activation_layer(attn_logits)

        return attn, attn_logits


class ChannelAttentionModule(nn.Module):  # type: ignore
    """Channel Attention Module."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        down_ratio: int = 16,
        inplace: bool = True,
    ) -> None:
        """Init."""
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // down_ratio, kernel_size=1, bias=False),
            bn_relu(out_ch // down_ratio, inplace=inplace),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch // down_ratio, out_ch, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sequential(nn.Sigmoid())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        initialize_weights(self.conv1, self.conv2)

    def forward(
        self, high_level_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        feats_avg = self.avg_pool(high_level_features)
        feats_max = self.max_pool(high_level_features)

        feats_max = self.conv1(feats_max)
        feats_avg = self.conv1(feats_avg)

        feats_max = self.conv2(feats_max)
        feats_avg = self.conv2(feats_avg)

        attn_logits = feats_max + feats_avg
        attn = self.sigmoid(attn_logits)

        return attn, attn_logits


def scale_space_rendering(
    scales: List[float], attn_logits: Dict[float, torch.Tensor]
) -> Dict[float, torch.Tensor]:
    """Scale space rendering attention."""
    attns = {}
    for i, scale in enumerate(scales):
        attns[scale] = 1 - torch.exp(-attn_logits[scale])

        if i > 0:
            for s in scales[:i]:
                attns[scale] *= torch.exp(-attn_logits[s])

    return attns


def make_attn_head(
    in_ch: int, out_ch: int, bot_ch: int = 256
) -> nn.Sequential:
    """HMA attention head."""
    od = OrderedDict(
        [
            (
                "conv0",
                nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
            ),
            ("bn0", nn.BatchNorm2d(bot_ch)),
            ("re0", nn.ReLU(inplace=True)),
        ]
    )

    od["conv1"] = nn.Conv2d(
        bot_ch, bot_ch, kernel_size=3, padding=1, bias=False
    )
    od["bn1"] = nn.BatchNorm2d(bot_ch)
    od["re1"] = nn.ReLU(inplace=True)

    od["conv2"] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od["sig"] = nn.Sigmoid()

    attn_head = nn.Sequential(od)
    return attn_head
