"""AFA-DLA-OCR seg model with SSR."""
from typing import Iterator, List, Tuple

import torch
from torch import nn
from torch.nn import Parameter

from afa.utils.structures import ArgsType, DictStrAny

from .afa_dla_seg_ssr import AFADLASegSSR
from .necks import OCRBlock
from .utils.attn_utils import SpatialAttentionModule, scale_space_rendering
from .utils.dla_utils import dla_resize
from .utils.ops import resize_x, scale_as


class AFADLAOCRSegSSR(AFADLASegSSR):
    """AFA DLA-Up-OCR segmentaion model with Scale-Space Rendering."""

    def __init__(
        self,
        *args: ArgsType,
        ocr_mid_channels: int = 256,
        ocr_key_channels: int = 128,
        ocr_aux_loss_weight: float = 0.4,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.ocr_aux_loss_weight = ocr_aux_loss_weight
        # remove original seg head
        delattr(self, "fc")
        delattr(self, "up")

        # scale-space attention across scales
        self.scale_space_attn = SpatialAttentionModule(
            ocr_mid_channels,
            1,
            ocr_mid_channels,
            activation="abs",
        )

        # OCR
        self.ocr = OCRBlock(
            self.num_classes,
            self.high_level_ch,
            ocr_mid_channels,
            ocr_key_channels,
        )

    def _fwd(  # type: ignore
        self, input_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Single scale forward."""
        final_layers, final_feats = self.backbone(input_x)

        y, ocr_aux_pred, ocr_feats = self.ocr(final_feats)

        return y, ocr_feats, final_layers, ocr_aux_pred

    def nscale_training_forward(
        self, inputs: DictStrAny, scales: List[float]
    ) -> DictStrAny:
        """Training with n scales."""
        assert "gts" in inputs
        x_inputs = inputs["images"]

        x_1x = dla_resize(x_inputs, self.training)

        scales = sorted(scales)

        preds = {}
        ocr_aux_preds = {}
        feats = {}
        final_layers = {}
        attn_logits = {}

        (
            preds[1.0],
            feats[1.0],
            final_layers[1.0],
            ocr_aux_preds[1.0],
        ) = self._fwd(x_1x)

        preds[1.0] = scale_as(preds[1.0], x_inputs)
        ocr_aux_preds[1.0] = scale_as(ocr_aux_preds[1.0], x_inputs)

        for i, scale in enumerate(scales):
            if scale != 1.0:
                (
                    preds[scale],
                    feats[scale],
                    final_layers[scale],
                    ocr_aux_preds[scale],
                ) = self._fwd(resize_x(x_1x, scale))

                preds[scale] = scale_as(preds[scale], preds[1.0])
                ocr_aux_preds[scale] = scale_as(
                    ocr_aux_preds[scale], ocr_aux_preds[1.0]
                )
                feats[scale] = scale_as(feats[scale], feats[1.0])

            _, attn_logits[scale] = self.scale_space_attn(feats[scale])

        attns = scale_space_rendering(scales, attn_logits)

        aux_preds = {}
        losses = {}
        pred = 0
        ocr_aux_pred = 0
        for i, scale in enumerate(scales):
            attns[scale] = scale_as(attns[scale], preds[1.0])

            pred += preds[scale] * attns[scale]
            ocr_aux_pred += ocr_aux_preds[scale] * attns[scale]
            losses[
                f"scale_{scale}".replace(".", "_")
            ] = self.dla_aux_loss_weight * self.criterion(
                preds[scale], inputs["gts"], do_rmi=False
            )

            for j in range(4):
                up = getattr(self, "up_" + str(j + 1))
                fc = getattr(self, "fc_" + str(j + 1))

                aux_pred = self.softmax(up(fc(final_layers[scale][j])))
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

        losses["ocr_aux_loss"] = self.ocr_aux_loss_weight * self.criterion(
            ocr_aux_pred, inputs["gts"], do_rmi=False
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

        preds[1.0], feats[1.0], _, _ = self._fwd(x_1x)
        preds[1.0] = scale_as(preds[1.0], x_inputs)

        for scale in scales:
            if scale != 1.0:
                preds[scale], feats[scale], _, _ = self._fwd(
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
            # "pred_05": preds[0.5],
            # "attn_05": attns[0.5],
            # "pred_10": preds[1.0],
            # "attn_10": attns[1.0],
        }

        return output_dict

    def optim_parameters(self) -> Iterator[Parameter]:
        """Yield optim parameters."""
        for param in self.backbone.optim_parameters():
            yield param
        for param in self.scale_space_attn.parameters():
            yield param
        for param in self.ocr.parameters():
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
    return AFADLAOCRSegSSR(
        num_classes=num_classes,
        base_name="dla34",
        criterion=criterion,
        pretrained_base="imagenet",
        n_scales_training=n_scales_training,
        n_scales_inference=n_scales_inference,
    )


def dla102x(
    num_classes: int,
    criterion: nn.Module,
    n_scales_training: List[float],
    n_scales_inference: List[float],
) -> nn.Module:
    """Backbone as DLA-X-102."""
    return AFADLAOCRSegSSR(
        num_classes=num_classes,
        base_name="dla102x",
        criterion=criterion,
        pretrained_base="imagenet",
        n_scales_training=n_scales_training,
        n_scales_inference=n_scales_inference,
    )


def dla169(
    num_classes: int,
    criterion: nn.Module,
    n_scales_training: List[float],
    n_scales_inference: List[float],
) -> nn.Module:
    """Backbone as DLA-169."""
    return AFADLAOCRSegSSR(
        num_classes=num_classes,
        base_name="dla169",
        criterion=criterion,
        pretrained_base="imagenet",
        n_scales_training=n_scales_training,
        n_scales_inference=n_scales_inference,
    )
