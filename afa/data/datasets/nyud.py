"""NYUD dataset."""
from os import path

from pytorch_lightning.utilities.rank_zero import rank_zero_info

from afa.utils.structures import ArgsType

from .base_edge import BaseEdgeDataset
from .custom import make_dataset_folder


class NYUDv2(BaseEdgeDataset):
    """NYUDv2 dataset class."""

    def __init__(
        self,
        *args: ArgsType,
        nyud_input_type: str = "image",
        **kwargs: ArgsType,
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.color_mapping = [0, 0, 0, 255, 255, 255]
        self.root = path.join(self.asset_dir, "data/NYUD")

        if self.folder is not None:
            self.images = [d[0] for d in make_dataset_folder(self.folder)]
        else:
            if self.mode == "train":
                nyu_train_path = path.join(
                    self.root, f"{nyud_input_type}-train.lst"
                )
                with open(nyu_train_path, "r", encoding="utf-8") as f:
                    ynu_train_list = [
                        tuple(line.rstrip().split(" "))
                        for line in f.readlines()
                    ]
                self.images = [
                    path.join(self.root, t[0]) for t in ynu_train_list
                ]
                self.masks = [
                    path.join(self.root, t[1]) for t in ynu_train_list
                ]
                assert len(self.images) == len(self.masks)
            else:
                nyu_test_path = path.join(
                    self.root, f"{nyud_input_type}-test.lst"
                )
                with open(nyu_test_path, "r", encoding="utf-8") as f:
                    nyu_test_list = [
                        path.join(self.root, line.rstrip())
                        for line in f.readlines()
                    ]
                self.images = nyu_test_list
            rank_zero_info(f"mode {self.mode} found {len(self.images)} images")
