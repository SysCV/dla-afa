"""DLA-Up."""
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from afa.utils.structures import NDArrayI64

from ..utils.dla_utils import Identity, fill_up_weights, init_weights
from ..utils.mynn import bn_relu


class IDAUp(nn.Module):  # type: ignore
    """IDA-Up module."""

    def __init__(
        self,
        node_kernel: int,
        out_dim: int,
        channels: List[int],
        up_factors: NDArrayI64,
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

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(
                    out_dim * 2,
                    out_dim,
                    kernel_size=node_kernel,
                    stride=1,
                    padding=node_kernel // 2,
                    bias=False,
                ),
                bn_relu(out_dim),
            )
            setattr(self, "node_" + str(i), node)

        init_weights(self.modules())

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
            node = getattr(self, "node_" + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):  # type: ignore
    """DLA-Up module."""

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
                f"ida_{i}",
                IDAUp(
                    3,
                    channels[j],
                    in_channels[j:],
                    scales_array[j:] // scales_array[j],
                ),
            )
            scales_array[j + 1 :] = scales_array[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        """Forward."""
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, f"ida_{i}")
            x, y = ida(layers[-i - 2 :])
            layers[-i - 1 :] = y
        return x
