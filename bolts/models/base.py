from __future__ import annotations
from abc import abstractmethod
from typing import Mapping

from kit import implements
from kit.torch.data import TrainingMode
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from bolts.common import LRScheduler, MetricDict, Stage
from bolts.data import NamedSample

__all__ = ["ModelBase"]


class ModelBase(pl.LightningModule):

    _target_name: str | None

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
    def target_name(self) -> str:
        assert self._target_name is not None
        return self._target_name

    @target_name.setter
    def target_name(self, value: str) -> None:
        self._target_name = value

    @abstractmethod
    def _inference_step(self, batch: NamedSample, stage: Stage) -> STEP_OUTPUT:
        ...

    @abstractmethod
    def _inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> MetricDict:
        ...

    @implements(pl.LightningModule)
    def validation_step(self, batch: NamedSample, batch_idx: int) -> STEP_OUTPUT:
        return self._inference_step(batch=batch, stage="validate")

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results_dict = self._inference_epoch_end(outputs=outputs, stage="validate")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def test_step(self, batch: NamedSample, batch_idx: int) -> STEP_OUTPUT:
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results_dict = self._inference_epoch_end(outputs=outputs, stage="test")
        self.log_dict(results_dict)
