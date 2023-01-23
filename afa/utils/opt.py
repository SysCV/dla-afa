"""Args for AFA."""
from argparse import ArgumentParser, FileType, Namespace

import pytorch_lightning as pl
import yaml


def config_parser() -> Namespace:
    """Args parser for AFA."""
    parser = ArgumentParser(description="AFA parser.")
    parser.add_argument(
        "action",
        type=str,
        choices=["fit", "test", "predict"],
        help="Action to execute",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="custom images folder",
    )

    # System
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="checkpoint path",
    )
    parser.add_argument(
        "--asset_dir",
        type=str,
        default="asset_dirs",
        help="asset directory",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="work_dirs",
        help="experiment work directory",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="AFA",
        help="experiment name",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="experiment version",
    )
    parser.add_argument(
        "--inplace_abn",
        type=bool,
        default=True,
        help="use inplace-abn",
    )

    # Inference
    parser.add_argument(
        "--default_scale",
        type=float,
        default=1.0,
        help="default scale to run validation",
    )
    parser.add_argument(
        "--multi_scale_inference",
        action="store_true",
        help="multi scale inference using Avg. Pooling",
    )
    parser.add_argument(
        "--extra_scales",
        type=str,
        default=None,
        help="scales for multi scale inference using Avg. Pooling",
    )
    parser.add_argument(
        "--do_flip",
        action="store_true",
        help="do flip for inference",
    )
    parser.add_argument(
        "--seg_fix",
        action="store_true",
        help="use segfix for cityscapes",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="cityscapes",
        help="cityscapes, mapillary, bdd100k, bsds, nyud",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="specify dataset mode",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=None,
        help="cross-validation split for cityscapes id to use",
    )
    parser.add_argument(
        "--only_fine",
        action="store_true",
        help="only use fine data for cityscapes",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=2,
        help="cpu worker threads per dataloader instance",
    )
    parser.add_argument(
        "--class_uniform_tile",
        type=int,
        default=1024,
        help="tile size for class uniform sampling",
    )
    parser.add_argument(
        "--max_cu_epochs",
        type=int,
        default=150,
        help="class uniform max epochs",
    )
    parser.add_argument(
        "--coarse_boost_classes",
        type=str,
        default=None,
        help="use coarse annotations for specific classes",
    )
    parser.add_argument(
        "--color_aug",
        type=float,
        default=0.25,
        help="level of color augmentation",
    )
    parser.add_argument(
        "--gblur",
        action="store_true",
        help="use Guassian Blur Augmentation",
    )
    parser.add_argument(
        "--pre_size",
        type=int,
        default=None,
        help="resize the long edge of the image to this before augmentation",
    )
    parser.add_argument(
        "--crop_size",
        type=str,
        default="512,1024",
        help="training crop size: h,w",
    )
    parser.add_argument(
        "--scale_min",
        type=float,
        default=0.5,
        help="dynamically scale training images down to this size",
    )
    parser.add_argument(
        "--scale_max",
        type=float,
        default=2.0,
        help="dynamically scale training images up to this size",
    )
    parser.add_argument(
        "--random_rotate",
        type=int,
        default=10,
        help="degree of random rotate",
    )
    parser.add_argument(
        "--is_edge",
        action="store_true",
        help="whether is edge dataset",
    )
    parser.add_argument(
        "--nyud_input_type",
        type=str,
        default=None,
        help="Input type of NYUDv2 dataset.",
    )
    parser.add_argument(
        "--bsds_with_pascal",
        action="store_true",
        help="BSDS500 dataset with PASCAL VOC.",
    )

    # Model
    parser.add_argument(
        "--n_scales_inference",
        type=str,
        default=None,
        help="n scales inference",
    )
    parser.add_argument(
        "--n_scales_training",
        type=str,
        default=None,
        help="n scales training",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="afa_dla_up.dla34",
        help="network architecture",
    )
    parser.add_argument(
        "--samples_per_gpu",
        type=int,
        default=2,
        help="Batch size for training per gpu",
    )

    # Learning rate
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="base learning rate",
    )
    parser.add_argument(
        "--warmup_iters",
        type=float,
        default=-1,
        help="learning rate warm-up iteration",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.01,
        help="learning rate warm-up ratio",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="poly",
        help="lr schedule: poly or step",
    )
    parser.add_argument(
        "--poly_exp",
        type=float,
        default=1.0,
        help="polynomial LR exponent",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-4,
        help="minimum lr",
    )
    parser.add_argument(
        "--step_epochs",
        type=str,
        default=None,
        help="step LR epochs",
    )
    parser.add_argument(
        "--step_ratio",
        type=float,
        default=0.1,
        help="step LR ratio",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer",
    )
    parser.add_argument(
        "--amsgrad",
        action="store_true",
        default=False,
        help="amsgrad for adam",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay for optimizer",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for optimizer",
    )

    # Loss
    parser.add_argument(
        "--ocr_aux_loss_rmi",
        action="store_true",
        help="allow rmi for aux loss",
    )
    parser.add_argument(
        "--ocr_alpha",
        type=float,
        default=0.4,
        help="set HRNet OCR auxiliary loss weight",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="pos weight for loss",
    )

    # Visualization
    parser.add_argument(
        "--dump_assets",
        action="store_true",
        help="dump interesting assets",
    )
    parser.add_argument(
        "--dump_for_submission",
        action="store_true",
        help="dump assets for submission",
    )

    # Misc
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Get model detailed information",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Using wandb",
    )
    parser.add_argument(
        "--checkpoint_period",
        type=int,
        default=1,
        help="Period of saving checkpoint",
    )
    parser.add_argument(
        "--pbar_refresh_rate",
        type=int,
        default=20,
        help="Refresh rate of the progress bar",
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        type=FileType(mode="r"),
        help="YAML config file",
    )
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.config_file:
        config = yaml.safe_load(args.config_file)["config"]
        delattr(args, "config_file")
        arg_dict = args.__dict__
        for key, value in config.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    return args
