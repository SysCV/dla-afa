"""AFA optimizer."""
from argparse import Namespace

from torch import nn
from torch.optim import SGD, Adam, Optimizer


def build_optimizer(net: nn.Module, args: Namespace) -> Optimizer:
    """Build optimizer."""
    param_groups = net.optim_parameters()

    if args.optimizer == "sgd":
        optimizer = SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif args.optimizer == "adam":
        optimizer = Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    else:
        raise ValueError(f"Not a valid optimizer {args.optim_type}")
    return optimizer
