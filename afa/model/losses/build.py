"""Losses utils."""
from typing import Tuple

from torch import nn

from .cross_entropy import CrossEntropyLoss2d
from .rmi import RMILoss


def build_loss(
    ignore_label: int,
    num_classes: int,
    pos_weight: float,
    is_edge: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    """Build loss."""
    criterion = RMILoss(
        num_classes=num_classes,
        ignore_index=ignore_label,
        pos_weight=pos_weight,
        is_edge=is_edge,
    )

    criterion_val = CrossEntropyLoss2d(ignore_index=ignore_label)
    return criterion, criterion_val
