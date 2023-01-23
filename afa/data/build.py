"""AFA data utils."""
from argparse import Namespace

from .datasets import build_dataset
from .module import AFADataModule


def build_datamodule(args: Namespace) -> AFADataModule:
    """Build datamodule."""
    if args.action == "fit":
        mode = "train"
    elif args.folder is not None:
        mode = "test"
    elif args.mode is not None:
        mode = args.mode
    else:
        mode = "val"

    crop_size = [int(x) for x in args.crop_size.split(",")]

    train_set, val_set = build_dataset(
        dataset_name=args.dataset,
        mode=mode,
        cv=args.cv,
        only_fine=args.only_fine,
        class_uniform_tile=args.class_uniform_tile,
        asset_dir=args.asset_dir,
        crop_size=crop_size,
        scale_max=args.scale_max,
        scale_min=args.scale_min,
        random_rotate=args.random_rotate,
        color_aug=args.color_aug,
        gblur=args.gblur,
        pre_size=args.pre_size,
        folder=args.folder,
        nyud_input_type=args.nyud_input_type,
        bsds_with_pascal=args.bsds_with_pascal,
    )

    return AFADataModule(
        workers_per_gpu=args.workers_per_gpu,
        samples_per_gpu=args.samples_per_gpu,
        train_set=train_set,
        val_set=val_set,
        dump_for_submission=args.dump_for_submission,
        dump_assets=args.dump_assets,
        is_edge=args.is_edge,
    )
