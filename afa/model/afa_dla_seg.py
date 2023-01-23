"""AFA-DLA seg model."""
from typing import Iterator, Optional

from torch import nn
from torch.nn import Parameter

from afa.utils.structures import DictStrAny

from .necks import AFADLAUpTrunk
from .utils.dla_utils import dla_resize, dla_seg_head, init_weights
from .utils.ops import scale_as


class AFADLASeg(nn.Module):  # type: ignore
    """AFA DLA-Up segmentaion model."""

    def __init__(
        self,
        base_name: str,
        num_classes: int,
        criterion: nn.Module,
        pretrained_base: Optional[str] = None,
        down_ratio: int = 2,
        dla_aux_loss_weight: float = 0.05,
    ) -> None:
        """Init."""
        super().__init__()

        self.criterion = criterion
        self.num_classes = num_classes
        self.backbone = AFADLAUpTrunk(
            base_name, pretrained_base=pretrained_base, down_ratio=down_ratio
        )
        self.dla_aux_loss_weight = dla_aux_loss_weight

        # seg head
        self.fc, self.up, self.softmax = dla_seg_head(
            self.backbone.first_level, num_classes, self.backbone.channels
        )

        # Aux seg head
        self.fc_1, self.up_1, _ = dla_seg_head(
            self.backbone.first_level + 3, num_classes, self.backbone.channels
        )
        self.fc_2, self.up_2, _ = dla_seg_head(
            self.backbone.first_level + 2, num_classes, self.backbone.channels
        )
        self.fc_3, self.up_3, _ = dla_seg_head(
            self.backbone.first_level + 1, num_classes, self.backbone.channels
        )
        self.fc_4, self.up_4, _ = dla_seg_head(
            self.backbone.first_level, num_classes, self.backbone.channels
        )

        init_weights(self.fc.modules())
        init_weights(self.fc_1.modules())
        init_weights(self.fc_2.modules())
        init_weights(self.fc_3.modules())
        init_weights(self.fc_4.modules())

    def forward(self, inputs: DictStrAny) -> DictStrAny:
        """Forward."""
        x_inputs = inputs["images"]

        x_1x = dla_resize(x_inputs, self.training)

        final_layers, final_feats = self.backbone(x_1x)

        pred = self.softmax(self.up(self.fc(final_feats)))

        pred = scale_as(pred, x_inputs)

        if self.training:
            losses = {}
            for i, final_layer in enumerate(final_layers):
                up = getattr(self, f"up_{i+1}")
                fc = getattr(self, f"fc_{i+1}")

                aux_pred = self.softmax(up(fc(final_layer)))
                aux_pred = scale_as(aux_pred, pred)

                losses[
                    f"dla_aux_loss_{i}"
                ] = self.dla_aux_loss_weight * self.criterion(
                    aux_pred, inputs["gts"], do_rmi=False
                )

            losses["main_loss"] = self.criterion(pred, inputs["gts"])
            return losses

        return {"pred": pred}

    def optim_parameters(self) -> Iterator[Parameter]:
        """Yield optim parameters."""
        for param in self.backbone.optim_parameters():
            yield param
        for param in self.fc.parameters():
            yield param
        for param in self.fc_1.parameters():
            yield param
        for param in self.fc_2.parameters():
            yield param
        for param in self.fc_3.parameters():
            yield param
        for param in self.fc_4.parameters():
            yield param


def dla34(num_classes: int, criterion: nn.Module) -> nn.Module:
    """Backbone as DLA-34."""
    return AFADLASeg(
        "dla34", num_classes, criterion=criterion, pretrained_base="imagenet"
    )


def dla102x(num_classes: int, criterion: nn.Module) -> nn.Module:
    """Backbone as DLA-X-102."""
    return AFADLASeg(
        "dla102x", num_classes, criterion=criterion, pretrained_base="imagenet"
    )


def dla169(num_classes: int, criterion: nn.Module) -> nn.Module:
    """Backbone as DLA-169."""
    return AFADLASeg(
        "dla169", num_classes, criterion=criterion, pretrained_base="imagenet"
    )
