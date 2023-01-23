"""Semantic segmentation evaluator for AFA.

# Code adapted from:
# https://github.com/NVIDIA/semantic-segmentation/blob/main/utils/trnval_utils.py # pylint: disable=line-too-long

Source License
# Copyright 2020 Nvidia Corporation

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this # pylint: disable=line-too-long
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tabulate import tabulate

from afa.data.datasets import BaseDataset
from afa.utils.structures import ArgsType, DictStrAny, NDArrayF32


class MeanIoUCallback(Callback):
    """Mean IoU metric callback."""

    def __init__(
        self,
        dataset: BaseDataset,
    ) -> None:
        """Init."""
        self.dataset_name = dataset.name
        self.dataset_mode = dataset.mode
        self.dataset_root = dataset.root
        self.num_classes = dataset.num_classes

        self._iou_acc: List[NDArrayF32] = []
        self._val_loss: List[NDArrayF32] = []

        self.best_record = {
            "epoch": -1,
            "val_loss": -1,
            "acc": 0,
            "acc_cls": 0,
            "fwavacc": 0,
            "mean_iu": 0,
        }

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._iou_acc = []
        self._val_loss = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: ArgsType,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Wait for on_test_batch_end PL hook to call 'process'."""
        _, gts, _, _ = batch

        iou_acc = fast_hist(
            outputs["predictions"].flatten(),  # type: ignore
            gts.to("cpu").numpy().flatten(),
            self.num_classes,
        )
        self._iou_acc.append(iou_acc)
        self._val_loss.append(outputs["val_loss"])  # type: ignore

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_validation_epoch_end PL hook to call 'evaluate'."""
        iou_acc = sum(self._iou_acc)
        iou_acc_sum = pl_module.all_gather(torch.tensor(iou_acc))
        val_loss = sum(self._val_loss)
        val_loss_sum = pl_module.all_gather(torch.tensor(val_loss))
        if trainer.is_global_zero:
            self.evaluate(iou_acc_sum, val_loss_sum, pl_module)
        self.reset()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: ArgsType,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Wait for on_test_batch_end PL hook to call 'process'."""
        _, gts, _, _ = batch

        iou_acc = fast_hist(
            outputs["predictions"].flatten(),  # type: ignore
            gts.to("cpu").numpy().flatten(),
            self.num_classes,
        )
        self._iou_acc.append(iou_acc)
        self._val_loss.append(outputs["val_loss"])  # type: ignore

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_test_epoch_end PL hook to call 'evaluate'."""
        iou_acc = sum(self._iou_acc)
        iou_acc_sum = pl_module.all_gather(torch.tensor(iou_acc))
        val_loss = sum(self._val_loss)
        val_loss_sum = pl_module.all_gather(torch.tensor(val_loss))
        if trainer.is_global_zero:
            self.evaluate(iou_acc_sum, val_loss_sum, pl_module)

    def evaluate(
        self,
        iou_acc_sum: torch.Tensor,
        val_loss_sum: torch.Tensor,
        pl_module: pl.LightningModule,
    ) -> None:
        """Mean IoU evaluation."""
        iou_acc_sum = iou_acc_sum.sum(axis=0)
        val_loss_sum = val_loss_sum.sum(axis=0) / len(pl_module.val_obj)
        metrics = self.eval_metrics(
            iou_acc_sum.cpu().numpy(),
            val_loss_sum.cpu().numpy(),
            pl_module.current_epoch,
            log_dir=pl_module.log_dir,
            trainid_to_name=pl_module.val_obj.trainid_to_name,
        )
        pl_module.log(
            "val/mIoU",
            torch.tensor(metrics["mean_iu"]).float(),
            rank_zero_only=True,
        )
        pl_module.log("val/loss", val_loss_sum, rank_zero_only=True)

    def eval_metrics(
        self,
        hist: NDArrayF32,
        val_loss: NDArrayF32,
        epoch: int,
        log_dir: str,
        trainid_to_name: Dict[int, str],
    ) -> DictStrAny:
        """Modified IoU mechanism for on-the-fly IoU calculations."""
        iu, acc, acc_cls = calculate_iou(hist)

        print_str_list = []
        print_str_list.append(
            print_evaluate_results(hist, iu, id2cat=trainid_to_name)
        )

        freq = hist.sum(axis=1) / hist.sum()
        mean_iu = np.nanmean(iu)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        metrics = {
            "val_loss": val_loss,
            "acc": acc,
            "acc_cls": acc_cls,
            "fwavacc": fwavacc,
            "mean_iu": mean_iu,
        }

        print_str_list.append(f"Mean: {(mean_iu * 100):2.2f}")

        if mean_iu > self.best_record["mean_iu"]:
            self.best_record["epoch"] = epoch
            self.best_record["val_loss"] = val_loss  # type: ignore
            self.best_record["acc"] = acc  # type: ignore
            self.best_record["acc_cls"] = acc_cls  # type: ignore
            self.best_record["fwavacc"] = fwavacc
            self.best_record["mean_iu"] = mean_iu

        print_str_list.append("-" * 110)
        print_str_list.append(
            f"this : [epoch {epoch}], [val_loss {val_loss:0.5f}], "
            + f"[acc {acc:0.5f}], "
            + f"[acc_cls {acc_cls:.5f}], "
            + f"[fwawacc {fwavacc:.5f}], "
            + f"[mean_iu {mean_iu:.5f}]"
        )
        print_str_list.append(
            f"best : [epoch {self.best_record['epoch']}], "
            + f"[val_loss {self.best_record['val_loss']:0.5f}], "
            + f"[acc {self.best_record['acc']:0.5f}], "
            + f"[acc_cls {self.best_record['acc_cls']:.5f}], "
            + f"[fwawacc {self.best_record['fwavacc']:.5f}], "
            + f"[mean_iu {self.best_record['mean_iu']:.5f}]"
        )
        print_str_list.append("-" * 110)
        print_str = "\n".join(print_str_list)
        rank_zero_info(print_str)

        # Metrics file
        with open(
            f"{log_dir}/metrics.csv", mode="a+", encoding="utf-8"
        ) as metrics_fp:
            metrics_writer = csv.writer(metrics_fp, delimiter=",")

            csv_line = ["epoch"]
            csv_line.append(str(epoch))

            for k, v in metrics.items():
                csv_line.append(k)
                csv_line.append(v)

            metrics_writer.writerow(csv_line)
            metrics_fp.flush()

        return metrics


def fast_hist(
    pred: NDArrayF32, gtruth: NDArrayF32, num_classes: int
) -> NDArrayF32:
    """Compute hist.

    stretch ground truth labels by num_classes
      class 0  -> 0
      class 1  -> 19
      class 18 -> 342

    TP at 0 + 0, 1 + 1, 2 + 2 ...

    TP exist where value == num_classes * class_id + class_id
    FP = col[class].sum() - TP
    FN = row[class].sum() - TP
    """
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)

    hist = np.bincount(
        num_classes * gtruth[mask].astype(int) + pred[mask],
        minlength=num_classes**2,
    )
    hist = hist.reshape(num_classes, num_classes)
    return hist  # type: ignore


def calculate_iou(
    hist_data: NDArrayF32,
) -> Tuple[NDArrayF32, NDArrayF32, NDArrayF32]:
    """Calculate IoU."""
    acc = np.diag(hist_data).sum() / hist_data.sum()
    acc_cls = np.diag(hist_data) / hist_data.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    divisor = (
        hist_data.sum(axis=1) + hist_data.sum(axis=0) - np.diag(hist_data)
    )
    iu = np.diag(hist_data) / divisor
    return iu, acc, acc_cls


def print_evaluate_results(
    hist: NDArrayF32, iu: NDArrayF32, id2cat: Dict[int, str]
) -> str:
    """Print evaluation results."""
    iu_fp = hist.sum(axis=0) - np.diag(hist)
    iu_fn = hist.sum(axis=1) - np.diag(hist)
    iu_tp = np.diag(hist)  # type: ignore

    header = ["Id", "label"]
    header.extend(["IoU", "TP", "FP", "FN", "Precision", "Recall"])

    tabulate_data = []

    for class_id, _iu in enumerate(iu):
        class_data = []
        class_data.append(class_id)
        class_name = f"{id2cat[class_id]}" if class_id in id2cat else ""
        class_data.append(class_name)  # type: ignore
        class_data.append(_iu * 100)

        total_pixels = hist.sum()
        class_data.append(100 * iu_tp[class_id] / total_pixels)
        class_data.append(100 * iu_fp[class_id] / total_pixels)
        class_data.append(100 * iu_fn[class_id] / total_pixels)
        if (iu_tp[class_id] + iu_fp[class_id]) > 0:
            precision = iu_tp[class_id] / (iu_tp[class_id] + iu_fp[class_id])
        else:
            precision = 0
        class_data.append(precision)
        if (iu_tp[class_id] + iu_fn[class_id]) > 0:
            recall = iu_tp[class_id] / (iu_tp[class_id] + iu_fn[class_id])
        else:
            recall = 0
        class_data.append(recall)
        tabulate_data.append(class_data)

    print_str = "IoU:\n" + str(
        tabulate((tabulate_data), headers=header, floatfmt="1.2f")
    )
    return print_str
