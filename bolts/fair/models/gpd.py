"""Zhang Gradient Projection Debiasing Baseline Model."""
from __future__ import annotations
from typing import Mapping, NamedTuple

import ethicml as em
from kit import implements
from kit.torch import CrossEntropyLoss, TrainingMode
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
import torchmetrics
from torchmetrics import MetricCollection

from bolts.data.structures import TernarySample
from bolts.models.base import ModelBase
from bolts.structures import Stage

__all__ = ["Gpd"]


def compute_proj_grads(*, model: nn.Module, loss_p: Tensor, loss_a: Tensor, alpha: float) -> None:
    """Computes the adversarial gradient projection term.

    Args:
        model (nn.Module): Model whose parameters the gradients are to be computed w.r.t.
        loss_p (Tensor): Prediction loss.
        loss_a (Tensor): Adversarial loss.
        alpha (float): Pre-factor for adversarial loss.
    """
    grad_p = torch.autograd.grad(loss_p, tuple(model.parameters()), retain_graph=True)
    grad_a = torch.autograd.grad(loss_a, tuple(model.parameters()), retain_graph=True)

    def _proj(a: Tensor, b: Tensor) -> Tensor:
        return b * torch.sum(a * b) / torch.sum(b * b).clamp(min=torch.finfo(b.dtype).eps)

    grad_p = [p - _proj(p, a) - alpha * a for p, a in zip(grad_p, grad_a)]

    for param, grad in zip(model.parameters(), grad_p):
        param.grad = grad


def compute_grad(*, model: nn.Module, loss: Tensor) -> None:
    """Computes the adversarial gradient projection term.

    Args:
        model (nn.Module): Model whose parameters the gradients are to be computed w.r.t.
        loss (Tensor): Adversarial loss.
    """
    grad_list = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)

    for param, grad in zip(model.parameters(), grad_list):
        param.grad = grad


class GpdOut(NamedTuple):
    s: Tensor
    y: Tensor


class Gpd(ModelBase):
    """Zhang Mitigating Unwanted Biases."""

    def __init__(
        self,
        *,
        adv: nn.Module,
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

        self.automatic_optimization = False  # Mark for manual optimization

    @implements(ModelBase)
    def _inference_epoch_end(
        self, outputs: list[Mapping[str, Tensor]], stage: Stage
    ) -> dict[str, Tensor]:
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

        results_dict = {
            f"{stage}/acc_{label}": self.accs[f"{stage.name}_{label}"].compute()
            for label in ("s", "y")
        }
        results_dict.update({f"{stage}/{k}": v for k, v in results.items()})
        return results_dict

    def _get_losses(
        self, model_out: GpdOut, *, batch: TernarySample
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_s = batch.s.view(-1, 1).float()
        loss_adv = self._loss_adv_fn(model_out.s, target=target_s)
        target_y = batch.y.view(-1, 1).float()
        loss_clf = self._loss_clf_fn(model_out.y, target=target_y)
        return loss_adv, loss_clf, loss_adv + loss_clf

    @implements(pl.LightningModule)
    def training_step(self, batch: TernarySample, batch_idx: int) -> None:
        opt = self.optimizers()
        opt.zero_grad()

        model_out: GpdOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out=model_out, batch=batch)

        logs = {
            f"{Stage.fit}/adv_loss": loss_adv.item(),
            f"{Stage.fit}/clf_loss": loss_clf.item(),
            f"{Stage.fit}/loss": loss.item(),
        }
        compute_proj_grads(model=self.enc, loss_p=loss_clf, loss_a=loss_adv, alpha=1.0)
        compute_grad(model=self.adv, loss=loss_adv)
        compute_grad(model=self.clf, loss=loss_clf)

        for _label in ("s", "y"):
            tm_acc = self.accs[f"{Stage.fit.name}_{_label}"]
            _target = getattr(batch, _label).view(-1).long()
            _acc = tm_acc(getattr(model_out, _label).argmax(-1), _target)
            logs.update({f"train/acc_{_label}": _acc})
        self.log_dict(logs)
        opt.step()

        if (self.lr_sched_interval is TrainingMode.step) and (
            self.global_step % self.lr_sched_freq == 0
        ):
            sch = self.lr_schedulers()
            sch.step()
        if (self.lr_sched_interval is TrainingMode.epoch) and self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()

    @implements(ModelBase)
    def _inference_step(self, batch: TernarySample, *, stage: Stage) -> dict[str, Tensor]:
        model_out: GpdOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out=model_out, batch=batch)
        logs = {
            f"{stage}/loss": loss.item(),
            f"{stage}/loss_adv": loss_adv.item(),
            f"{stage}/loss_clf": loss_clf.item(),
        }

        for _label in ("s", "y"):
            tm_acc = self.accs[f"{stage.name}_{_label}"]
            _target = getattr(batch, _label).view(-1).long()
            _acc = tm_acc(getattr(model_out, _label).argmax(-1), _target)
            logs.update({f"{stage}/acc_{_label}": _acc})
        self.log_dict(logs)
        return {
            "y": batch.y.view(-1),
            "s": batch.s.view(-1),
            "preds": model_out.y.sigmoid().round().squeeze(-1),
        }

    def reset_parameters(self) -> None:
        """Reset the models."""
        self.apply(self._maybe_reset_parameters)

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    @implements(nn.Module)
    def forward(self, x: Tensor) -> GpdOut:
        z = self.enc(x)
        y = self.clf(z)
        s = self.adv(z)
        return GpdOut(s=s, y=y)
