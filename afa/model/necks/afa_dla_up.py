"""AFA DLA-Up."""
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from afa.utils.structures import NDArrayI64

from ..backbones import dla
from ..utils.attn_utils import ChannelAttentionModule, SpatialAttentionModule
from ..utils.dla_utils import Identity, fill_up_weights, init_weights
from ..utils.mynn import bn_relu
from ..utils.ops import scale_as


class AFAIDAUp(nn.Module):  # type: ignore
    """AFA IDA-Up module."""

    def __init__(
        self, out_dim: int, channels: List[int], up_factors: NDArrayI64
    ) -> None:
        """Init."""
        super().__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim, kernel_size=1, stride=1, bias=False),
                    bn_relu(out_dim),
                )
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim,
                    out_dim,
                    f * 2,
                    stride=f,
                    padding=f // 2,
                    output_padding=0,
                    groups=out_dim,
                    bias=False,
                )
                fill_up_weights(up)
            setattr(self, f"proj_{i}", proj)
            setattr(self, f"up_{i}", up)

        init_weights(self.modules())

        for i in range(1, len(channels)):
            sa = SpatialAttentionModule(
                out_dim, 1, out_dim, activation="sigmoid"
            )
            setattr(self, f"sa_{i}", sa)
            ca = ChannelAttentionModule(out_dim, out_dim)
            setattr(self, f"ca_{i}", ca)

    def forward(
        self, layers: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward."""
        assert len(self.channels) == len(
            layers
        ), f"{len(self.channels)} vs {len(layers)} layers"
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, f"up_{i}")
            project = getattr(self, f"proj_{i}")
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            # spatial attention
            sa = getattr(self, f"sa_{i}")
            spatial_attn, _ = sa(x)

            # channel attention
            ca = getattr(self, f"ca_{i}")
            channel_attn, _ = ca(layers[i])

            x = (
                spatial_attn * (1 - channel_attn) * x
                + (1 - spatial_attn) * channel_attn * layers[i]
            )
            y.append(x)
        return x, y


class AFADLAUp(nn.Module):  # type: ignore
    """AFA DLA-Up module."""

    def __init__(
        self,
        channels: List[int],
        scales: List[int],
        in_channels: Optional[List[int]] = None,
    ) -> None:
        """Init."""
        super().__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales_array: NDArrayI64 = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                f"afa_ida_{i}",
                AFAIDAUp(
                    channels[j],
                    in_channels[j:],
                    scales_array[j:] // scales_array[j],
                ),
            )
            scales_array[j + 1 :] = scales_array[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward."""
        final_layers = []
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, f"afa_ida_{i}")
            x, y = ida(layers[-i - 2 :])
            layers[-i - 1 :] = y
            final_layers.append(x)
        return final_layers


class AFADLAUpTrunk(nn.Module):  # type: ignore
    """AFA DLA-Up trunk module."""

    def __init__(
        self,
        base_name: str,
        pretrained_base: Optional[str] = None,
        down_ratio: int = 2,
    ) -> None:
        """Init."""
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.base = dla.__dict__[base_name](
            pretrained=pretrained_base, return_levels=True
        )

        self.channels = self.base.channels
        scales = [
            2**i for i in range(len(self.channels[self.first_level :]))
        ]
        self.high_level_ch = self.channels[self.first_level]

        self.afa_dla_up = AFADLAUp(
            self.channels[self.first_level :], scales=scales
        )

        # spatial attention across levels
        self.level_sa = SpatialAttentionModule(
            self.high_level_ch, 1, self.high_level_ch, activation="sigmoid"
        )

        # channel attention across levels
        self.level_ca = ChannelAttentionModule(
            self.high_level_ch, self.high_level_ch
        )

        self.proj_1 = nn.Sequential(
            nn.Conv2d(
                self.channels[self.first_level + 2],
                self.high_level_ch,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            bn_relu(self.high_level_ch),
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2d(
                self.channels[self.first_level + 1],
                self.high_level_ch,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            bn_relu(self.high_level_ch),
        )

        init_weights(self.proj_1.modules())
        init_weights(self.proj_2.modules())

    def final_feature_fusion(
        self, final_layers: List[torch.Tensor]
    ) -> torch.Tensor:
        """Final feature fusion."""
        feats = {}
        spatial_attns = {}
        channel_attns = {}
        final_feats = 0

        for i, _ in enumerate(final_layers):
            if i < len(final_layers) - 1:
                up = getattr(self, f"proj_{i+1}")
                feats[i + 1] = scale_as(up(final_layers[i]), final_layers[-1])
            else:
                feats[i + 1] = final_layers[i]

        for i, _ in enumerate(final_layers):
            spatial_attn, _ = self.level_sa(feats[i + 1])
            channel_attn, _ = self.level_ca(feats[i + 1])

            spatial_attns[i + 1] = spatial_attn
            channel_attns[i + 1] = channel_attn

            for j in range(1, i + 1):
                spatial_attns[j] = spatial_attns[j] * (1 - spatial_attn)
                channel_attns[j] = channel_attns[j] * (1 - channel_attn)

        for i in range(len(final_layers)):
            final_feats = (
                final_feats
                + spatial_attns[i + 1] * channel_attns[i + 1] * feats[i + 1]
            )

        return final_feats

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward."""
        x = self.base(feat)

        final_layers = self.afa_dla_up(x[self.first_level :])

        final_feats = self.final_feature_fusion(final_layers[1:])

        return final_layers, final_feats

    def optim_parameters(self) -> Iterator[Parameter]:
        """Yield optim parameters."""
        for param in self.base.parameters():
            yield param
        for param in self.afa_dla_up.parameters():
            yield param
        for param in self.level_sa.parameters():
            yield param
        for param in self.level_ca.parameters():
            yield param
        for param in self.proj_1.parameters():
            yield param
        for param in self.proj_2.parameters():
            yield param
