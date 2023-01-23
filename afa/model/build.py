"""Build model."""
import importlib
from argparse import Namespace
from typing import Optional

from torch import nn

from afa.data.datasets import BaseDataset

from .losses import build_loss
from .module import AFASegmentaionModel


def get_net(
    arch: str,
    num_classes: int,
    criterion: nn.Module,
    n_scales_training: Optional[str] = None,
    n_scales_inference: Optional[str] = None,
) -> nn.Module:
    """Get network architecture."""
    network = f"afa.model.{arch}"
    module = network[: network.rfind(".")]
    model = network[network.rfind(".") + 1 :]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)

    if n_scales_training is not None or n_scales_inference is not None:
        n_scales_training = [float(x) for x in n_scales_training.split(",")]  # type: ignore # pylint: disable=line-too-long
        n_scales_inference = [float(x) for x in n_scales_inference.split(",")]  # type: ignore # pylint: disable=line-too-long
        net = net_func(
            num_classes=num_classes,
            criterion=criterion,
            n_scales_training=n_scales_training,
            n_scales_inference=n_scales_inference,
        )
    else:
        net = net_func(num_classes=num_classes, criterion=criterion)
    return net


def build_lightningmodule(
    args: Namespace,
    train_obj: Optional[BaseDataset],
    val_obj: BaseDataset,
    log_dir: str,
) -> AFASegmentaionModel:
    """Build model."""
    criterion, criterion_val = build_loss(
        num_classes=val_obj.num_classes,
        ignore_label=val_obj.ignore_label,
        pos_weight=args.pos_weight,
        is_edge=args.is_edge,
    )
    net = get_net(
        args.arch,
        val_obj.num_classes,
        criterion,
        args.n_scales_training,
        args.n_scales_inference,
    )

    return AFASegmentaionModel(
        args, net, criterion_val, train_obj, val_obj, log_dir
    )
