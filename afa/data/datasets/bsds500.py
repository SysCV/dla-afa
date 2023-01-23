"""BSDS500 dataset."""
from os import path

import numpy as np
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from afa.utils.structures import ArgsType, NDArrayF32

from .base_edge import BaseEdgeDataset
from .custom import make_dataset_folder


class BSDS500(BaseEdgeDataset):
    """BSDS500 dataset class."""

    def __init__(
        self,
        *args: ArgsType,
        threshold: float = 0.3,
        bsds_with_pascal: bool = False,
        **kwargs: ArgsType,
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.color_mapping = [0, 0, 0, 255, 255, 255]
        self.threshold = threshold
        self.root = path.join(self.asset_dir, "data/BSDS500")
        hed_dir = path.join(self.root, "HED-BSDS")

        if self.folder is not None:
            self.images = [d[0] for d in make_dataset_folder(self.folder)]
        else:
            if self.mode == "train":
                hed_train_path = path.join(hed_dir, "train_pair.lst")
                with open(hed_train_path, "r", encoding="utf-8") as f:
                    hed_train_list = [
                        tuple(line.rstrip().split(" "))
                        for line in f.readlines()
                    ]
                self.images = [
                    path.join(hed_dir, t[0]) for t in hed_train_list
                ]
                self.masks = [path.join(hed_dir, t[1]) for t in hed_train_list]

                if bsds_with_pascal:
                    pascal_dir = path.join(self.root, "PASCAL")
                    pascal_train_path = path.join(pascal_dir, "train_pair.lst")
                    with open(pascal_train_path, "r", encoding="utf-8") as f:
                        pascal_train_list = [
                            tuple(line.rstrip().split(" "))
                            for line in f.readlines()
                        ]
                    for t in pascal_train_list:
                        self.images.append(path.join(pascal_dir, t[0]))
                        self.masks.append(path.join(pascal_dir, t[1]))
                assert len(self.images) == len(self.masks)
            else:
                hed_test_path = path.join(hed_dir, "test.lst")
                with open(hed_test_path, "r", encoding="utf-8") as f:
                    hed_test_list = [
                        path.join(hed_dir, line.rstrip())
                        for line in f.readlines()
                    ]
                self.images = hed_test_list
            rank_zero_info(f"mode {self.mode} found {len(self.images)} images")

    def _get_boundaries_target(  # type: ignore
        self,
        filepath: str,
    ) -> Image.Image:
        """Get boundaries target."""
        target: NDArrayF32 = np.array(Image.open(filepath), dtype=np.float32)
        if len(target.shape) == 2:
            target = target / 255.0
        else:
            target = target[:, :, 0] / 255.0
        target[target > self.threshold] = 1
        target[np.logical_and(target > 0, target < self.threshold)] = 2
        target = Image.fromarray(target.astype(np.uint8))
        return target
