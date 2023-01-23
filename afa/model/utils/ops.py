"""AFA operations."""
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F


def flip_tensor(input_x: torch.Tensor, dim: int) -> torch.Tensor:
    """Flip input tensor along the dimension."""
    dim = input_x.dim() + dim if dim < 0 else dim
    return input_x[
        tuple(
            slice(None, None)
            if i != dim
            else torch.arange(input_x.size(i) - 1, -1, -1).long()
            for i in range(input_x.dim())
        )
    ]


def resize_tensor(
    inputs: torch.Tensor, target_size: Union[List[int], Tuple[int, int]]
) -> torch.Tensor:
    """Resize input tensor to target size."""
    inputs = F.interpolate(
        inputs,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return inputs


def scale_as(input_x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale input to the same size as target."""
    target_size = target.size(2), target.size(3)

    return resize_tensor(input_x, target_size)


def resize_x(input_x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Scale input by factor."""
    return F.interpolate(
        input_x,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=True,
    )
