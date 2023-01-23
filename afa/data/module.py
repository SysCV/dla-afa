"""AFA data module."""
from typing import Optional, no_type_check

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .callbacks import ImageDumperCallback, MeanIoUCallback
from .datasets import BaseDataset
from .sampler import DistributedSampler


class AFADataModule(pl.LightningDataModule):
    """Data modeul for AFA."""

    def __init__(
        self,
        workers_per_gpu: int,
        samples_per_gpu: int,
        val_set: BaseDataset,
        train_set: Optional[BaseDataset] = None,
        dump_for_submission: bool = False,
        dump_assets: bool = False,
        is_edge: bool = False,
    ) -> None:
        """Init."""
        super().__init__()  # type: ignore
        self.dataset_name = val_set.name
        self.workers_per_gpu = workers_per_gpu
        self.samples_per_gpu = samples_per_gpu
        self.train_set = train_set
        self.val_set = val_set
        self.dump_for_submission = dump_for_submission
        self.dump_assets = dump_assets
        self.is_edge = is_edge

    @no_type_check
    def setup(self, stage: Optional[str] = None) -> None:
        """Data preparation operations to perform on every GPU."""
        if stage == "predict":
            self.trainer.callbacks += [
                ImageDumperCallback(
                    dataset=self.val_set,
                    output_dir=self.trainer.log_dir,
                    dump_for_submission=self.dump_for_submission,
                    dump_assets=self.dump_assets,
                    is_edge=self.is_edge,
                    dump_gts=self.val_set.mode != "test",
                )
            ]
        elif not self.is_edge:
            self.trainer.callbacks += [MeanIoUCallback(dataset=self.val_set)]

        # pylint: disable=protected-access
        self.trainer._callback_connector._attach_model_logging_functions()
        self.trainer.callbacks = (
            self.trainer._callback_connector._reorder_callbacks(
                self.trainer.callbacks
            )
        )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        train_sampler = DistributedSampler(  # type: ignore
            self.train_set,
            shuffle=True,
        )
        return DataLoader(
            self.train_set,
            batch_size=self.samples_per_gpu,
            num_workers=self.workers_per_gpu,
            shuffle=False,
            drop_last=True,
            sampler=train_sampler,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        val_sampler = DistributedSampler(  # type: ignore
            self.val_set,
            shuffle=False,
        )
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.workers_per_gpu,
            shuffle=False,
            drop_last=False,
            sampler=val_sampler,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        test_sampler = DistributedSampler(  # type: ignore
            self.val_set,
            shuffle=False,
        )
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.workers_per_gpu,
            shuffle=False,
            drop_last=False,
            sampler=test_sampler,
        )

    def predict_dataloader(self) -> DataLoader:
        """Prediction dataloader."""
        predict_sampler = DistributedSampler(  # type: ignore
            self.val_set,
            shuffle=False,
        )
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.workers_per_gpu,
            shuffle=False,
            drop_last=False,
            sampler=predict_sampler,
        )
