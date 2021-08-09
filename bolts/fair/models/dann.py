"""DANN (Domain Adversarial Neural Network) model."""
from __future__ import annotations
from typing import Mapping, NamedTuple

import ethicml as em
from kit import implements
from kit.torch import CrossEntropyLoss, TrainingMode
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor, autograd, nn
import torchmetrics
from torchmetrics import MetricCollection

from bolts.data.structures import TernarySample
from bolts.models.base import ModelBase
from bolts.structures import MetricDict, Stage

__all__ = ["Dann"]


class DannOut(NamedTuple):
    s: Tensor
    y: Tensor


class GradReverse(autograd.Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx: autograd.Function, x: Tensor, lambda_: float) -> Tensor:
        """Do GRL."""
        if lambda_ < 0:
            raise ValueError(f"Argument 'lambda_' to GradReverse.forward must be non-negative.")
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: Tensor) -> tuple[Tensor, Tensor | None]:
        """Reverse (and optionally scale) the gradient."""
        return -ctx.lambda_ * grad_output, None


def grad_reverse(features: Tensor, *, lambda_: float = 1.0) -> Tensor:
    """Gradient Reversal layer."""
    return GradReverse.apply(features, lambda_)


class Dann(ModelBase):
    """Ganin's Domain Adversarial NN."""

    def __init__(
        self,
        *,
        adv: nn.Module,
        enc: nn.Module,
        clf: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        grl_lambda: float = 1.0,
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
        self.grl_lambda = grl_lambda
        self.learning_rate = lr
        self.weight_decay = weight_decay

        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq

        self.adv = adv
        self.enc = enc
        self.clf = clf

        self._loss_adv_fn = CrossEntropyLoss()
        self._loss_clf_fn = CrossEntropyLoss()

        self.accs = MetricCollection(
            {
                f"{stage.name}_{label}": torchmetrics.Accuracy()
                for stage in Stage
                for label in ("s", "y")
            }
        )

    def _get_losses(
        self, model_out: DannOut, *, batch: TernarySample
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_s = batch.s.view(-1, 1).float()
        loss_adv = self._loss_adv_fn(model_out.s, target=target_s)
        target_y = batch.y.view(-1, 1).float()
        loss_clf = self._loss_clf_fn(model_out.y, target=target_y)
        return loss_adv, loss_clf, loss_adv + loss_clf

    @implements(pl.LightningModule)
    def training_step(self, batch: TernarySample, batch_idx: int) -> Tensor:
        model_out: DannOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out=model_out, batch=batch)

        logs = {
            f"{Stage.fit.value}/adv_loss": loss_adv.item(),
            f"{Stage.fit.value}": loss_clf.item(),
            f"{Stage.fit.value}/loss": loss.item(),
        }
        for _label in ("s", "y"):
            tm_acc = self.accs[f"{Stage.fit.name}_{_label}"]
            _target = getattr(batch, _label).view(-1).long()
            _acc = tm_acc(getattr(model_out, _label).argmax(-1), _target)
            logs.update({f"{Stage.fit.value}/acc_{_label}": _acc})
        self.log_dict(logs)
        return loss

    @implements(ModelBase)
    def _inference_epoch_end(self, outputs: list[Mapping[str, Tensor]], stage: Stage) -> MetricDict:
        all_y = torch.cat([output_step["y"] for output_step in outputs], 0)
        all_s = torch.cat([output_step["s"] for output_step in outputs], 0)
        all_preds = torch.cat([output_step["preds"] for output_step in outputs], 0)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(all_s, dtype=torch.float).detach().cpu().numpy(), columns=["x0"]
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

        results_dict = {
            f"{stage.value}/acc_{label}": self.accs[f"{stage.name}_{label}"].compute()
            for label in ("s", "y")
        }
        results_dict.update({f"{stage.value}/{k}": v for k, v in results.items()})
        return results_dict

    @implements(ModelBase)
    def _inference_step(self, batch: TernarySample, *, stage: Stage) -> STEP_OUTPUT:
        model_out: DannOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out=model_out, batch=batch)
        logs = {
            f"{stage.value}/loss": loss.item(),
            f"{stage.value}/loss_adv": loss_adv.item(),
            f"{stage.value}/loss_clf": loss_clf.item(),
        }

        for _label in ("s", "y"):
            tm_acc = self.accs[f"{stage.name}_{_label}"]
            _target = getattr(batch, _label).view(-1).long()
            _acc = tm_acc(getattr(model_out, _label).argmax(-1), _target)
            logs.update({f"{stage.value}/acc_{_label}": _acc})
        self.log_dict(logs)
        return {
            "y": batch.y.view(-1),
            "s": batch.s.view(-1),
            "preds": model_out.y.sigmoid().round().squeeze(-1),
        }

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the models."""
        self.apply(self._maybe_reset_parameters)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> DannOut:
        z = self.enc(x)
        y = self.clf(z)
        s = self.adv(grad_reverse(z, lambda_=self.grl_lambda))
        return DannOut(s=s, y=y)
