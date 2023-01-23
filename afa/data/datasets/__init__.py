"""AFA Dataset Init."""
from .base import BaseDataset
from .build import build_dataset
from .cityscapes_labels import label2trainid as cityscapes_label2trainid

__all__ = [
    "BaseDataset",
    "cityscapes_label2trainid",
    "build_dataset",
]
