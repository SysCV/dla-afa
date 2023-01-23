"""Build AFA model."""
from typing import Callable, List, Optional, Tuple, no_type_check

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torchmetrics import MeanMetric

from afa.data.datasets import BaseDataset
from afa.utils.structures import ArgsType, DictStrAny, NDArrayF32, NDArrayI64

from .optimize import build_optimizer, build_scheduler
from .utils.ops import flip_tensor, resize_tensor
from .utils.trnval_utils import parse_output


class AFASegmentaionModel(pl.LightningModule):
    """AFA Segmentaion model."""

    def __init__(
        self,
        args: ArgsType,
        net: nn.Module,
        criterion_val: nn.Module,
        train_obj: Optional[BaseDataset],
        val_obj: BaseDataset,
        log_dir: str,
    ):
        """Init."""
        super().__init__()
        self.save_hyperparameters(args, ignore=["net", "criterion_val"])

        self.net = net
        self.criterion_val = criterion_val
        self.train_obj = train_obj
        self.val_obj = val_obj
        self.log_dir = log_dir

        self.dataset_name = args.dataset
        self.is_edge = args.is_edge
        self.max_cu_epochs = args.max_cu_epochs
        self.optim = build_optimizer(net, args)
        self.schd = build_scheduler(self.optim, args)
        self.warmup_iters = args.warmup_iters
        self.warmup_ratio = args.warmup_ratio
        self.seg_fix = args.seg_fix

        self.scales = [args.default_scale]
        if args.multi_scale_inference:
            self.scales.extend(
                [float(x) for x in args.extra_scales.split(",")]
            )
            self.scales = sorted(self.scales)

            rank_zero_info(
                "Using multi-scale inference (AVGPOOL) with "
                + f"scales {self.scales}"
            )

        if args.do_flip:
            rank_zero_info("Do flip during inference")
            self.flips = [1, 0]
        else:
            self.flips = [0]

        if self.seg_fix:
            rank_zero_info("Run Seg-Fix post-processing")

    def on_train_epoch_start(self) -> None:
        """Train epoch start."""
        assert self.train_obj is not None
        if self.dataset_name == "cityscapes":
            if (
                self.train_obj.class_uniform_pct
                and self.current_epoch == self.max_cu_epochs
            ):
                self.train_obj.disable_coarse()
        if self.dataset_name not in ["bsds500", "nyud"]:
            self.train_obj.build_epoch()

    def training_step(  # type: ignore # pylint: disable=unused-argument
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[float]],
        batch_idx: int,
    ):
        """Training loop."""
        images, gts, _, _ = batch
        inputs = {"images": images, "gts": gts}
        losses = self.net(inputs)
        losses["loss"] = sum(list(losses.values()))

        log_dict = {}
        metric_attributes = []
        for k, v in losses.items():
            if not hasattr(self, k):
                metric = MeanMetric()
                metric.to(self.device)
                setattr(self, k, metric)

            metric = getattr(self, k)
            metric(v.detach())
            log_dict["train/" + k] = metric
            metric_attributes += [k]

        for (k, v), k_name in zip(log_dict.items(), metric_attributes):
            self.log(
                k,
                v,
                logger=True,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                metric_attribute=k_name,
            )
        return losses

    def forward_test(
        self,
        input_images: torch.Tensor,
        gts: torch.Tensor,
        image_names: List[str],
    ) -> Tuple[NDArrayI64, DictStrAny, Optional[NDArrayF32]]:
        """Pure inference."""
        input_size = input_images.size(2), input_images.size(3)

        output = 0.0
        for flip in self.flips:
            for scale in self.scales:
                if flip == 1:
                    images = flip_tensor(input_images, 3)
                else:
                    images = input_images

                infer_size = [round(sz * scale) for sz in input_size]

                if scale != 1.0:
                    images = resize_tensor(images, infer_size)

                inputs = {"images": images}

                output_dict = self.net(inputs)
                _pred = output_dict["pred"]

                if scale != 1.0:
                    _pred = resize_tensor(_pred, input_size)

                if flip == 1:
                    output = output + flip_tensor(_pred, 3)
                else:
                    output = output + _pred

        output = output / len(self.scales) / len(self.flips)

        predictions, assets = parse_output(
            dataset_name=self.dataset_name,
            ignore_label=self.val_obj.ignore_label,
            image_names=image_names,
            output_dict=output_dict,
            output=output,
            gts=gts,
            dataset_mode=self.val_obj.mode,
            dataset_root=self.val_obj.root,
            do_seg_fix=self.seg_fix,
            is_edge=self.is_edge,
        )

        if self.val_obj.mode != "test":
            val_loss = self.criterion_val(output, gts).cpu().numpy()
        else:
            val_loss = None

        return predictions, assets, val_loss

    def validation_step(  # type: ignore # pylint: disable=unused-argument
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[float]],
        batch_idx: int,
    ) -> DictStrAny:
        """Validation loop."""
        input_images, gts, image_names, _ = batch
        predictions, _, val_loss = self.forward_test(
            input_images, gts, image_names
        )

        return {
            "predictions": predictions,
            "val_loss": val_loss,
        }

    def test_step(  # type: ignore # pylint: disable=unused-argument
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[float]],
        batch_idx: int,
    ) -> DictStrAny:
        """Test loop."""
        input_images, gts, image_names, _ = batch
        predictions, _, val_loss = self.forward_test(
            input_images, gts, image_names
        )

        return {
            "predictions": predictions,
            "val_loss": val_loss,
        }

    def predict_step(  # pylint: disable=unused-argument
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[float]],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> DictStrAny:
        """Predict loop."""
        input_images, gts, image_names, _ = batch
        predictions, assets, _ = self.forward_test(
            input_images, gts, image_names
        )

        return {
            "input_images": input_images,
            "gt_images": gts,
            "img_names": image_names,
            "predictions": predictions,
            "assets": assets,
        }

    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[lr_scheduler._LRScheduler]]:
        """Config optimizer and scheduler."""
        return [self.optim], [self.schd]

    @no_type_check
    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        """Optimizer step plus learning rate warmup."""
        base_lr = optimizer.defaults.get("lr", None)
        if base_lr is None:
            raise ValueError(
                "Couldn't determine base LR from optimizer defaults: "
                f"{optimizer.defaults}"
            )

        if self.trainer.global_step < self.warmup_iters:
            k = (1 - self.trainer.global_step / self.warmup_iters) * (
                1 - self.warmup_ratio
            )
            warmup_lr = base_lr * (1 - k)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        elif self.trainer.global_step == self.warmup_iters:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def on_load_checkpoint(
        self, checkpoint: DictStrAny
    ) -> None:  # pragma: no cover
        """Allow for mismatched shapes when loading checkpoints."""
        state_dict = checkpoint["state_dict"]
        model_state_dict: DictStrAny = self.state_dict()
        for k in model_state_dict:
            if k in checkpoint["state_dict"]:
                if (
                    checkpoint["state_dict"][k].shape
                    != model_state_dict[k].shape
                ):
                    rank_zero_info(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
            else:
                rank_zero_info(
                    f"Skip parameter: {k}, which is not in the checkpoint."
                )
                state_dict[k] = model_state_dict[k]
