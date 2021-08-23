from __future__ import annotations

from kit import implements
from kit.torch import CrossEntropyLoss, TrainingMode
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor, nn
import torch.nn as nn

from bolts.data import BinarySample
from bolts.types import Loss, MetricDict, Stage

from .base import ModelBase
from .utils import (
    accuracy,
    aggregate_over_epoch,
    make_no_grad,
    precision_at_k,
    prefix_keys,
)

__all__ = ["ClassifierERM", "FineTuner"]


class ClassifierERM(ModelBase):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
        loss_fn: Loss = CrossEntropyLoss(reduction="mean"),
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
        )
        self.model = model
        self.loss_fn = loss_fn

    @implements(pl.LightningModule)
    def training_step(self, batch: BinarySample, batch_idx: int) -> Tensor:
        logits = self.forward(batch.x)
        loss = self.loss_fn(input=batch.x, target=batch.y)
        results_dict = {
            "loss": loss.item(),
            "acc": accuracy(logits=logits, targets=batch.y),
        }
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.fit), sep="/")
        self.log_dict(results_dict)

        return loss

    @implements(ModelBase)
    def _inference_step(self, batch: BinarySample, stage: Stage) -> STEP_OUTPUT:
        logits = self.forward(batch.x)
        return {"logits": logits, "targets": batch.y}

    @implements(ModelBase)
    def _inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> MetricDict:
        logits_all = aggregate_over_epoch(outputs=outputs, metric="logits")
        targets_all = aggregate_over_epoch(outputs=outputs, metric="targets")
        acc1, acc5 = precision_at_k(logits=logits_all, target=targets_all, top_k=(1, 5))
        loss = self.loss_fn(input=logits_all, target=targets_all)
        return {'loss': loss, 'acc1': acc1, 'acc5': acc5}

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()  # type: ignore

    def reset_parameters(self) -> None:
        self.apply(self._maybe_reset_parameters)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class FineTuner(ClassifierERM):
    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
        loss_fn: Loss = CrossEntropyLoss(reduction="mean"),
    ) -> None:
        encoder = make_no_grad(encoder)
        model = nn.Sequential(encoder, classifier)
        super().__init__(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
            loss_fn=loss_fn,
        )
        self.encoder = encoder
        self.classifier = classifier
