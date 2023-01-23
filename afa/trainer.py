"""AFA trainer."""
import os.path as osp
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
from thop import profile

from .config import init_config
from .data import build_datamodule
from .model import build_lightningmodule
from .utils import (
    AFAProgressBar,
    config_parser,
    init_random_seed,
    is_torch_tf32_available,
    setup_logger,
)


def main() -> None:
    """Main Function."""
    args = config_parser()

    init_config(args)

    if is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if args.action == "fit":
        init_random_seed(args.seed)

    if args.version is None:
        timestamp = (
            str(datetime.now())
            .split(".", maxsplit=1)[0]
            .replace(" ", "_")
            .replace(":", "-")
        )
        args.version = timestamp

    output_dir = osp.join(args.work_dir, args.exp_name, args.version)

    setup_logger(osp.join(output_dir, f"log_{timestamp}.txt"))

    if args.wandb:
        exp_logger = pl.loggers.WandbLogger(
            save_dir=output_dir,
            project=args.exp_name,
            name=args.version,
        )
    else:
        exp_logger = pl.loggers.TensorBoardLogger(  # type:ignore
            save_dir=args.work_dir,
            name=args.exp_name,
            version=args.version,
            default_hp_metric=False,
        )

    lr_monitor_callbacks = pl.callbacks.LearningRateMonitor(
        logging_interval="step"
    )

    progressbar_callbacks = AFAProgressBar(refresh_rate=args.pbar_refresh_rate)

    checkpoint_callbacks = pl.callbacks.ModelCheckpoint(
        dirpath=osp.join(output_dir, "checkpoints"),
        verbose=True,
        save_last=True,
        every_n_epochs=args.checkpoint_period,
        save_on_train_epoch_end=True,
    )

    data_module = build_datamodule(args)

    model = build_lightningmodule(
        args,
        data_module.train_set,
        data_module.val_set,
        output_dir,
    )

    if args.summary:
        print(str(model.net))
        img = torch.randn(1, 3, 1024, 2048).cuda()
        flops, params = profile(model.net.cuda(), inputs=({"images": img},))
        print(f"FLOPs {flops/1000000000}G, params {params/1000000}M")
        sys.exit()

    args.logger = exp_logger
    args.callbacks = [
        lr_monitor_callbacks,
        progressbar_callbacks,
        checkpoint_callbacks,
    ]

    trainer = pl.Trainer.from_argparse_args(args)

    if args.action == "fit":
        trainer.fit(model, data_module, ckpt_path=args.weights)
    elif args.action == "test":
        trainer.test(model, data_module, ckpt_path=args.weights, verbose=False)
    elif args.action == "predict":
        trainer.predict(model, data_module, ckpt_path=args.weights)


if __name__ == "__main__":
    main()
