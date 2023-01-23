"""Build AFA dataset."""
from typing import List, Optional, Tuple

import torchvision.transforms as standard_transforms
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from afa.utils.structures import ListAny

from ..transforms import (
    ColorJitter,
    MaskToTensor,
    RandomCrop,
    RandomGaussianBlur,
    RandomHorizontallyFlip,
    RandomRotate,
    RandomSizeAndCrop,
    Scale,
)
from .base import BaseDataset
from .bdd100k import BDD100K
from .bsds500 import BSDS500
from .cityscapes import CityScapes
from .mapillary import MapillaryVistas
from .nyud import NYUDv2


def build_dataset(
    dataset_name: str,
    asset_dir: str,
    mode: str,
    cv: int,
    only_fine: bool,
    crop_size: List[int],
    class_uniform_tile: int,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    random_rotate: int = 10,
    color_aug: float = 0.25,
    gblur: bool = False,
    pre_size: Optional[int] = None,
    folder: Optional[str] = None,
    nyud_input_type: Optional[str] = None,
    bsds_with_pascal: Optional[bool] = None,
) -> Tuple[Optional[BaseDataset], BaseDataset]:
    """Get dataset class."""
    if dataset_name == "cityscapes":
        num_classes = 19
        ignore_label = 255
        dataset_cls = CityScapes
    elif dataset_name == "bdd100k":
        num_classes = 19
        ignore_label = 255
        dataset_cls = BDD100K
    elif dataset_name == "mapillary":
        num_classes = 65
        ignore_label = 65
        dataset_cls = MapillaryVistas
    elif dataset_name == "nyud":
        num_classes = 2
        ignore_label = 2
        dataset_cls = NYUDv2
    elif dataset_name == "bsds500":
        num_classes = 2
        ignore_label = 2
        dataset_cls = BSDS500
    else:
        raise NotImplementedError(f"{dataset_name} is not supported!")

    rank_zero_info(f"dataset = {dataset_name}")
    rank_zero_info(f"num_classes = {num_classes}")
    rank_zero_info(f"ignore_label = {ignore_label}")

    if dataset_name == "bdd100k":
        dataset_mean = [0.279, 0.293, 0.290]
        dataset_std = [0.247, 0.265, 0.276]
    else:
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]

    if pre_size is not None:
        rank_zero_info(
            f"Scale image such that longer side is equal to {pre_size}"
        )

    # Joint transformations that must happen on both image and mask
    train_joint_transform_list: ListAny = []

    if dataset_name == "bsds500":
        train_joint_transform_list.append(
            RandomCrop(
                crop_size=crop_size,
                ignore_index=ignore_label,
                nopad=False,
            )
        )
    else:
        train_joint_transform_list.append(
            RandomSizeAndCrop(
                crop_size=crop_size,
                ignore_index=ignore_label,
                nopad=False,
                scale_min=scale_min,
                scale_max=scale_max,
                pre_size=pre_size,
            )
        )
        if dataset_name != "nyud":
            train_joint_transform_list.append(RandomHorizontallyFlip())

            train_joint_transform_list.append(
                RandomRotate(angle=random_rotate)
            )

    # Image only augmentations
    train_input_transform: ListAny = []

    if nyud_input_type != "hha":
        train_input_transform += [
            ColorJitter(
                brightness=color_aug,
                contrast=color_aug,
                saturation=color_aug,
                hue=color_aug,
            )
        ]

    if gblur:
        train_input_transform += [RandomGaussianBlur()]

    mean_std = (dataset_mean, dataset_std)
    train_input_transform += [
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std),
    ]
    train_input_transform = standard_transforms.Compose(train_input_transform)
    target_train_transform = MaskToTensor()

    val_input_transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std),
        ]
    )
    target_transform = MaskToTensor()

    if dataset_name == "mapillary" and pre_size is not None:
        val_joint_transform_list = [Scale(pre_size)]
    else:
        val_joint_transform_list = None

    if mode == "train":
        train_set = dataset_cls(
            name=dataset_name,
            asset_dir=asset_dir,
            mode=mode,
            cv=cv,
            only_fine=only_fine,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mean=dataset_mean,
            std=dataset_std,
            class_uniform_tile=class_uniform_tile,
            joint_transform_list=train_joint_transform_list,
            img_transform=train_input_transform,
            label_transform=target_train_transform,
            nyud_input_type=nyud_input_type,
            bsds_with_pascal=bsds_with_pascal,
        )
        val_mode = "val"
    else:
        train_set = None
        val_mode = mode

    if dataset_name in ["nyud", "bsds500"]:
        val_mode = "test"

    val_set = dataset_cls(
        name=dataset_name,
        asset_dir=asset_dir,
        mode=val_mode,
        cv=cv,
        num_classes=num_classes,
        ignore_label=ignore_label,
        mean=dataset_mean,
        std=dataset_std,
        joint_transform_list=val_joint_transform_list,
        img_transform=val_input_transform,
        label_transform=target_transform,
        folder=folder,
        nyud_input_type=nyud_input_type,
        bsds_with_pascal=bsds_with_pascal,
    )
    return train_set, val_set
