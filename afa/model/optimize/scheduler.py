"""AFA lr scheduler."""
from argparse import Namespace
from typing import List

from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR


class PolyLRScheduler(lr_scheduler._LRScheduler):  # type: ignore # pylint: disable=protected-access,line-too-long
    """Polynomial learning rate decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_progress: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize PolyLRScheduler."""
        self.max_progress = max_progress
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute current learning rate."""
        if self._step_count >= self.max_progress:  # pragma: no cover
            return [self.min_lr for _ in self.base_lrs]
        coeff = (1 - self._step_count / self.max_progress) ** self.power
        return [
            (base_lr - self.min_lr) * coeff + self.min_lr
            for base_lr in self.base_lrs
        ]


def build_scheduler(
    optimizer: Optimizer, args: Namespace
) -> lr_scheduler._LRScheduler:
    """Build scheduler."""
    if args.lr_scheduler == "step":
        milestones = [int(step) for step in args.step_epochs.split(",")]
        scheduler = MultiStepLR(optimizer, milestones, gamma=args.step_ratio)
    elif args.lr_scheduler == "poly":
        scheduler = PolyLRScheduler(
            optimizer,
            max_progress=args.max_epochs,
            power=args.poly_exp,
            min_lr=args.min_lr,
        )
    else:
        raise ValueError(f"Not a valid lr scheduler {args.schd_type}")
    return scheduler
