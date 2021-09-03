from __future__ import annotations
from abc import abstractmethod
import inspect
from typing import List, Mapping, Tuple, cast

from kit import implements
from kit.misc import gcopy
from kit.torch.data import TrainingMode
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from bolts.data import NamedSample
from bolts.data.datamodules.base import PBDataModule
from bolts.types import LRScheduler, MetricDict, Stage

from .utils import prefix_keys

__all__ = ["PBModel"]


class PBModel(pl.LightningModule):
    def __init__(
        self,
        *,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq
        self._datamodule: PBDataModule | None = None
        self._trainer: pl.Trainer | None = None

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[Mapping[str, LRScheduler | int | TrainingMode]]]:
        opt = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer=opt, T_0=self.lr_initial_restart, T_mult=self.lr_restart_mult
            ),
            "interval": self.lr_sched_interval.name,
            "frequency": self.lr_sched_freq,
        }
        return [opt], [sched]

    @property
    def datamodule(self) -> PBDataModule:
        if self._datamodule is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.datamodule' cannot be accessed as '{cls_name}.build' has "
                "not yet been called.'"
            )
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule: PBDataModule) -> None:
        self._datamodule = datamodule
        self._datamodule.prepare_data()
        self._datamodule.setup()

    @property
    def trainer(self) -> pl.Trainer:
        if self._trainer is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.trainer' cannot be accessed as '{cls_name}.build' has "
                "not yet been called.'"
            )
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: pl.Trainer) -> None:
        self._trainer = trainer

    def build(self, datamodule: PBDataModule, *, trainer: pl.Trainer, copy: bool = True) -> None:
        if copy:
            datamodule = gcopy(datamodule, deep=False)
            trainer = gcopy(trainer, deep=True)
        self._datamodule = datamodule
        self._trainer = trainer
        self._build()
        # Retrieve all child models (attributes inheriting from PBModel)
        children = cast(
            List[Tuple[str, PBModel]],
            inspect.getmembers(self, lambda m: isinstance(m, PBModel)),
        )
        # Build all child models
        for _, child in children:
            child.build(datamodule=self.datamodule, trainer=self.trainer, copy=False)

    def _build(self) -> None:
        ...

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from the datamodule and trainer."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = self.datamodule.num_train_batches(drop_last=False)
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices  # type: ignore
        return (batches // effective_accum) * self.trainer.max_epochs

    @abstractmethod
    def inference_step(self, batch: NamedSample, stage: Stage) -> STEP_OUTPUT:
        ...

    @abstractmethod
    def inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> MetricDict:
        ...

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_step(self, batch: NamedSample, batch_idx: int) -> STEP_OUTPUT:
        return self.inference_step(batch=batch, stage=Stage.validate)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results_dict = self.inference_epoch_end(outputs=outputs, stage=Stage.validate)
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.validate), sep="/")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_step(self, batch: NamedSample, batch_idx: int) -> STEP_OUTPUT:
        return self.inference_step(batch=batch, stage=Stage.test)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results_dict = self.inference_epoch_end(outputs=outputs, stage=Stage.test)
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.test), sep="/")
        self.log_dict(results_dict)

    def fit(self) -> None:
        self.trainer.fit(model=self, datamodule=self.datamodule)

    def test(self, verbose: bool = True) -> None:
        self.trainer.test(model=self, datamodule=self.datamodule, verbose=verbose)

    def run(
        self,
        *,
        datamodule: PBDataModule,
        trainer: pl.Trainer,
        seed: int | None = None,
        copy: bool = True,
    ) -> None:
        """Seed, build, fit, and test the model."""
        pl.seed_everything(seed)
        self.build(datamodule=datamodule, trainer=trainer, copy=copy)
        self.fit()
        self.test()
