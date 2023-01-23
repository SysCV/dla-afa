"""BDD100K dataset."""
import itertools
from os import path

from pytorch_lightning.utilities.rank_zero import rank_zero_info

from afa.utils.structures import ArgsType

from ..uniform import build_centroids
from .base import BaseDataset
from .custom import make_dataset_folder

bddd100k_trainid_to_name = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffc light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


class BDD100K(BaseDataset):
    """BDD100K dataset class."""

    def __init__(
        self,
        *args: ArgsType,
        **kwargs: ArgsType,
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.root = path.join(self.asset_dir, "data/bdd100k")
        self.trainid_to_name = bddd100k_trainid_to_name
        self.fill_colormap()

        if self.folder is not None:
            self.all_imgs = make_dataset_folder(self.folder)
        else:
            img_root = path.join(self.root, "images/10k", self.mode)
            mask_root = path.join(
                self.root, "labels/sem_seg/colormaps", self.mode
            )
            self.all_imgs = self.find_images(
                img_root,
                mask_root,
                img_ext="jpg",
                mask_ext="png",
            )

            rank_zero_info(
                f"mode {self.mode} found {len(self.all_imgs)} images"
            )

        if self.mode == "train" and self.class_uniform_pct:
            self.centroids = build_centroids(
                dataset_name=self.name,
                imgs=self.all_imgs,
                num_classes=self.num_classes,
                centroid_root=self.centroid_root,
                class_uniform_tile=self.class_uniform_tile,
            )

        self.imgs = self.all_imgs  # type: ignore

    def fill_colormap(self) -> None:
        """Build colormap."""
        palette_array = [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]
        palette = list(itertools.chain.from_iterable(palette_array))
        zero_pad = 256 * 3 - len(palette)
        for _ in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette
