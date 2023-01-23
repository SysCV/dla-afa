"""DLA Seg model."""
from typing import Iterator, Optional

import numpy as np
from torch import nn
from torch.nn import Parameter

from afa.utils.structures import DictStrAny

from .backbones import dla
from .necks import DLAUp
from .utils.dla_utils import dla_resize, dla_seg_head, init_weights
from .utils.ops import scale_as


class DLASeg(nn.Module):  # type: ignore
    """DLA-Up segmentaion model."""

    def __init__(
        self,
        base_name: str,
        num_classes: int,
        criterion: nn.Module,
        pretrained_base: Optional[str] = None,
        down_ratio: int = 2,
    ) -> None:
        """Init."""
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.criterion = criterion
        self.first_level = int(np.log2(down_ratio))
        self.base = dla.__dict__[base_name](
            pretrained=pretrained_base, return_levels=True
        )
        channels = self.base.channels
        scales = [2**i for i in range(len(channels[self.first_level :]))]

        self.dla_up = DLAUp(channels[self.first_level :], scales=scales)

        # seg head
        self.fc, self.up, self.softmax = dla_seg_head(
            self.first_level, num_classes, channels
        )

        init_weights(self.fc.modules())

    def forward(self, inputs: DictStrAny) -> DictStrAny:
        """Forward."""
        x_inputs = inputs["images"]

        x_1x = dla_resize(x_inputs, self.training)

        x = self.base(x_1x)
        x = self.dla_up(x[self.first_level :])

        pred = self.softmax(self.up(self.fc(x)))
        pred = scale_as(pred, x_inputs)

        if self.training:
            loss = self.criterion(pred, inputs["gts"])

            return {"main_loss": loss}

        return {"pred": pred}

    def optim_parameters(self) -> Iterator[Parameter]:
        """Yield optim parameters."""
        for param in self.base.parameters():
            yield param
        for param in self.dla_up.parameters():
            yield param
        for param in self.fc.parameters():
            yield param


def dla34(num_classes: int, criterion: nn.Module) -> nn.Module:
    """Backbone as DLA-34."""
    return DLASeg(
        "dla34", num_classes, criterion=criterion, pretrained_base="imagenet"
    )


def dla102x(num_classes: int, criterion: nn.Module) -> nn.Module:
    """Backbone as DLA-X-102."""
    return DLASeg(
        "dla102x", num_classes, criterion=criterion, pretrained_base="imagenet"
    )


def dla169(num_classes: int, criterion: nn.Module) -> nn.Module:
    """Backbone as DLA-169."""
    return DLASeg(
        "dla169", num_classes, criterion=criterion, pretrained_base="imagenet"
    )
