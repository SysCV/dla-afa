"""Custom dataset."""
import os
from typing import List, Tuple

from pytorch_lightning.utilities.rank_zero import rank_zero_info


def make_dataset_folder(folder: str) -> List[Tuple[str, str]]:
    """Create Filename list for images in the provided path."""
    items = [(os.path.join(folder, f), "") for f in os.listdir(folder)]
    items = sorted(items)

    rank_zero_info(f"Found {len(items)} folder imgs")
    return items
