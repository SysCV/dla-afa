"""AFA-DLA seg model with SSR."""
from typing import Iterator, List, Tuple

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch import nn
from torch.nn import Parameter

from afa.utils.structures import ArgsType, DictStrAny

from .afa_dla_seg import AFADLASeg
from .utils.attn_utils import SpatialAttentionModule, scale_space_rendering
from .utils.dla_utils import dla_resize
from .utils.ops import resize_x, scale_as


class AFADLASegSSR(AFADLASeg):
    """AFA DLA-Up segmentaion model with Scale-Space Rendering."""

    def __init__(
        self,
        *args: ArgsType,
        n_scales_training: List[float],
        n_scales_inference: List[float],
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.n_scale_training = n_scales_training
        rank_zero_info("SSR Training with " + f"Scales {n_scales_training}")
        self.n_scales_inference = n_scales_inference
        rank_zero_info("SSR Inference with " + f"Scales {n_scales_inference}")
        self.high_level_ch = self.backbone.high_level_ch

        # Scale-space attention across scales
        self.scale_space_attn = SpatialAttentionModule(
            self.high_level_ch, 1, self.high_level_ch, activation="abs"
        )

    def _fwd(
        self, input_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Single scale forward."""
        final_layers, final_feats = self.backbone(input_x)

        y = self.softmax(self.up(self.fc(final_feats)))

        return y, final_feats, final_layers

    def nscale_training_forward(
        self, inputs: DictStrAny, scales: List[float]
    ) -> DictStrAny:
        """Training with n scales."""
        assert "gts" in inputs
        x_inputs = inputs["images"]

        x_1x = dla_resize(x_inputs, self.training)

        scales = sorted(scales)

        preds = {}
        feats = {}
        final_layers = {}
        attn_logits = {}

        preds[1.0], feats[1.0], final_layers[1.0] = self._fwd(x_1x)
        preds[1.0] = scale_as(preds[1.0], x_inputs)

        for i, scale in enumerate(scales):
            if scale != 1.0:
                preds[scale], feats[scale], final_layers[scale] = self._fwd(
                    resize_x(x_1x, scale)
                )

                preds[scale] = scale_as(preds[scale], preds[1.0])
                feats[scale] = scale_as(feats[scale], feats[1.0])

            _, attn_logits[scale] = self.scale_space_attn(feats[scale])

        attns = scale_space_rendering(scales, attn_logits)

        aux_preds = {}
        losses = {}
        pred = 0
        for i, scale in enumerate(scales):
            attns[scale] = scale_as(attns[scale], preds[1.0])

            pred += preds[scale] * attns[scale]
            losses[
                f"scale_{scale}".replace(".", "_")
            ] = self.dla_aux_loss_weight * self.criterion(
                preds[scale], inputs["gts"], do_rmi=False
            )

            for j in range(4):
                up = getattr(self, "up_" + str(j + 1))
                fc = getattr(self, "fc_" + str(j + 1))

                aux_pred = self.softmax(up(fc(final_layers[scale][j])))
                if scale != 1.0:
                    aux_pred = scale_as(aux_pred, preds[1.0])

                if i == 0:
                    aux_preds[j] = attns[scale] * aux_pred
                else:
                    aux_preds[j] += attns[scale] * aux_pred

        for i in range(4):
            losses[
                f"dla_aux_loss_{i}"
            ] = self.dla_aux_loss_weight * self.criterion(
                aux_preds[i], inputs["gts"], do_rmi=False
            )

        losses["main_loss"] = self.criterion(pred, inputs["gts"])

        return losses

    def nscale_forward(
        self, inputs: DictStrAny, scales: List[float]
    ) -> DictStrAny:
        """Inference with n scales."""
        x_inputs = inputs["images"]

        x_1x = dla_resize(x_inputs)

        scales = sorted(scales)

        preds = {}
        feats = {}
        attn_logits = {}

        preds[1.0], feats[1.0], _ = self._fwd(x_1x)
        preds[1.0] = scale_as(preds[1.0], x_inputs)

        for scale in scales:
            if scale != 1.0:
                preds[scale], feats[scale], _ = self._fwd(
                    resize_x(x_1x, scale)
                )

                preds[scale] = scale_as(preds[scale], preds[1.0])
                feats[scale] = scale_as(feats[scale], feats[1.0])

            _, attn_logits[scale] = self.scale_space_attn(feats[scale])

        attns = scale_space_rendering(scales, attn_logits)

        pred = 0
        for scale in scales:
            attns[scale] = scale_as(attns[scale], preds[1.0])
            pred += preds[scale] * attns[scale]

        pred = scale_as(pred, x_inputs)

        output_dict = {
            "pred": pred,
        }

        return output_dict

    def forward(self, inputs: DictStrAny) -> DictStrAny:
        """Forward."""
        if self.training:
            return self.nscale_training_forward(inputs, self.n_scale_training)

        return self.nscale_forward(inputs, self.n_scales_inference)

    def optim_parameters(self) -> Iterator[Parameter]:
        """Yield optim parameters."""
        for param in self.backbone.optim_parameters():
            yield param
        for param in self.scale_space_attn.parameters():
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


def dla34(
    num_classes: int,
    criterion: nn.Module,
    n_scales_training: List[float],
    n_scales_inference: List[float],
) -> nn.Module:
    """Backbone as DLA-34."""
    return AFADLASegSSR(
        num_classes=num_classes,
        base_name="dla34",
        criterion=criterion,
        pretrained_base="imagenet",
        n_scales_training=n_scales_training,
        n_scales_inference=n_scales_inference,
    )
