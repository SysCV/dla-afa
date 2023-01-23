"""CE loss."""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class CrossEntropyLoss2d(nn.Module):  # type: ignore
    """Cross Entroply NLL Loss."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
        reduction: str = "mean",
    ) -> None:
        """Init."""
        super().__init__()
        self.nll_loss = nn.NLLLoss(
            weight, reduction=reduction, ignore_index=ignore_index
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward."""
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
