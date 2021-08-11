"""ERM Baseline Model."""
from __future__ import annotations

import ethicml as em
from kit import implements
from kit.torch import CrossEntropyLoss, ReductionType, TrainingMode
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import Tensor, nn
import torchmetrics

from bolts.data.structures import TernarySample
from bolts.models import ModelBase
from bolts.structures import MetricDict, Stage

__all__ = ["ErmBaseline"]


class ErmBaseline(ModelBase):
    """Empirical Risk Minimisation baseline."""

    def __init__(
        self,
        *,
        enc: nn.Module,
        clf: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
        )
        self.enc = enc
        self.clf = clf
        self.net = nn.Sequential(self.enc, self.clf)

        self._target_name = "y"
        self._loss_fn = CrossEntropyLoss(reduction=ReductionType.mean)

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def _get_loss(self, logits: Tensor, *, batch: TernarySample) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y)

    @implements(pl.LightningModule)
    def training_step(self, batch: TernarySample, batch_idx: int) -> STEP_OUTPUT:
        logits = self.forward(batch.x)
        loss = self._get_loss(logits=logits, batch=batch)
        target = batch.y.view(-1).long()
        _acc = self.train_acc(logits.argmax(-1), target)
        self.log_dict(
            {
                f"train/loss": loss.item(),
                f"train/acc": _acc,
            }
        )
        return loss

    @implements(ModelBase)
    def _inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> MetricDict:
        all_y = torch.cat([step_output["y"] for step_output in outputs], 0)
        all_s = torch.cat([step_output["s"] for step_output in outputs], 0)
        all_preds = torch.cat([step_output["preds"] for step_output in outputs], 0)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(all_s, dtype=float).detach().cpu().numpy(), columns=["x0"]
            ),
            s=pd.DataFrame(all_s.detach().cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(all_y.detach().cpu().numpy(), columns=["y"]),
        )

        results = em.run_metrics(
            predictions=em.Prediction(hard=pd.Series(all_preds.argmax(-1).detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

        tm_acc = self.val_acc if stage is Stage.validate else self.test_acc
        results_dict = {f"{stage}/acc": tm_acc.compute().item()}
        results_dict.update({f"{stage}/{self.target_name}_{k}": v for k, v in results.items()})
        return results_dict

    @implements(ModelBase)
    def _inference_step(self, batch: TernarySample, *, stage: Stage) -> STEP_OUTPUT:
        logits = self.forward(batch.x)
        loss = self._get_loss(logits=logits, batch=batch)
        tm_acc = self.val_acc if stage is Stage.validate else self.test_acc
        target = batch.y.view(-1).long()
        _acc = tm_acc(logits.argmax(-1), target)
        self.log_dict(
            {
                f"{stage}/loss": loss.item(),
                f"{stage}/{self.target_name}_acc": _acc,
            }
        )
        return {
            "y": batch.y.view(-1),
            "s": batch.s.view(-1),
            "preds": logits.sigmoid().round().squeeze(-1),
        }

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()  # type: ignore

    def reset_parameters(self) -> None:
        """Reset the models."""
        self.enc.apply(self._maybe_reset_parameters)
        self.clf.apply(self._maybe_reset_parameters)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
