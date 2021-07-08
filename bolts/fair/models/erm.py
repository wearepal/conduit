"""ERM Baseline Model."""
from __future__ import annotations
from typing import Mapping

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchmetrics

from bolts.common import Stage
from bolts.fair.data.structures import DataBatch
from bolts.fair.losses import CrossEntropy
from bolts.fair.models.utils import LRScheduler, SchedInterval


class ErmBaseline(pl.LightningModule):
    """Empirical Risk Minimisation baseline."""

    def __init__(
        self,
        enc: nn.Module,
        clf: nn.Module,
        lr: float,
        weight_decay: float,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: SchedInterval = "epoch",
        lr_sched_freq: int = 1,
    ) -> None:
        super().__init__()
        self.enc = enc
        self.clf = clf
        self.net = nn.Sequential(self.enc, self.clf)

        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq

        self._target_name = "y"
        self._loss_fn = CrossEntropy(reduction="mean")

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    @property
    def target(self) -> str:
        return self._target_name

    @target.setter
    def target(self, target: str) -> None:
        self._target_name = target

    def _inference_epoch_end(
        self, output_results: list[Mapping[str, Tensor]], stage: Stage
    ) -> dict[str, Tensor]:
        all_y = torch.cat([_r["y"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_preds = torch.cat([_r["preds"] for _r in output_results], 0)

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

        tm_acc = self.val_acc if stage == "validate" else self.test_acc
        results_dict = {f"{stage}/acc": tm_acc.compute()}
        results_dict.update({f"{stage}/{self.target}_{k}": v for k, v in results.items()})
        return results_dict

    def _inference_step(self, batch: DataBatch, stage: Stage) -> dict[str, Tensor]:
        logits = self.forward(batch.x)
        loss = self._get_loss(logits, batch)
        tm_acc = self.val_acc if stage == "validate" else self.test_acc
        target = batch.y.view(-1).long()
        _acc = tm_acc(logits.argmax(-1), target)
        self.log_dict(
            {
                f"{stage}/loss": loss.item(),
                f"{stage}/{self.target}_acc": _acc,
            }
        )
        return {
            "y": batch.y.view(-1),
            "s": batch.s.view(-1),
            "preds": logits.sigmoid().round().squeeze(-1),
        }

    def _get_loss(self, logits: Tensor, batch: DataBatch) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y)

    def reset_parameters(self) -> None:
        """Reset the models."""
        self.enc.apply(self._maybe_reset_parameters)
        self.clf.apply(self._maybe_reset_parameters)

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[Mapping[str, LRScheduler | int | SchedInterval]]]:
        opt = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        sched = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer=opt, T_0=self.lr_initial_restart, T_mult=self.lr_restart_mult
            ),
            "interval": self.lr_sched_interval,
            "frequency": self.lr_sched_freq,
        }
        return [opt], [sched]

    @implements(pl.LightningModule)
    def test_epoch_end(self, output_results: list[Mapping[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(output_results=output_results, stage="test")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        logits = self.forward(batch.x)
        loss = self._get_loss(logits, batch)
        target = batch.y.view(-1).long()
        _acc = self.train_acc(logits.argmax(-1), target)
        self.log_dict(
            {
                f"train/loss": loss.item(),
                f"train/acc": _acc,
            }
        )
        return loss

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: list[Mapping[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(output_results=output_results, stage="validate")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="validate")

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
