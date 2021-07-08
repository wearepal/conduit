"""Zhang Gradient Projection Debiasing Baseline Model."""
from __future__ import annotations
from typing import Mapping, NamedTuple

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchmetrics
from torchmetrics import MetricCollection
from typing_inspect import get_args

from bolts.common import Stage
from bolts.fair.data import DataBatch
from bolts.fair.losses import CrossEntropy
from bolts.fair.models.utils import LRScheduler, SchedInterval

__all__ = ["Gpd"]


def compute_proj_grads(*, model: nn.Module, loss_p: Tensor, loss_a: Tensor, alpha: float) -> None:
    """Computes the adversarial gradient projection term.

    Args:
        model (nn.Module): Model whose parameters the gradients are to be computed w.r.t.
        loss_p (Tensor): Prediction loss.
        loss_a (Tensor): Adversarial loss.
        alpha (float): Pre-factor for adversarial loss.
    """
    grad_p = torch.autograd.grad(loss_p, model.parameters(), retain_graph=True)
    grad_a = torch.autograd.grad(loss_a, model.parameters(), retain_graph=True)

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
    grad_list = torch.autograd.grad(loss, model.parameters(), retain_graph=True)

    for param, grad in zip(model.parameters(), grad_list):
        param.grad = grad


class GpdOut(NamedTuple):
    s: Tensor
    y: Tensor


class Gpd(pl.LightningModule):
    """Zhang Mitigating Unwanted Biases."""

    def __init__(
        self,
        adv: nn.Module,
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
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq

        self.adv = adv
        self.enc = enc
        self.clf = clf

        self._loss_adv_fn = CrossEntropy()
        self._loss_clf_fn = CrossEntropy()

        self.accs = MetricCollection(
            {
                f"{stage}_{label}": torchmetrics.Accuracy()
                for stage in get_args(Stage)
                for label in ("s", "y")
            }
        )

        self.automatic_optimization = False  # Mark for manual optimization

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

        results_dict = {
            f"{stage}/acc_{label}": self.accs[f"{stage}_{label}"].compute() for label in ("s", "y")
        }
        results_dict.update({f"{stage}/{k}": v for k, v in results.items()})
        return results_dict

    def _get_losses(self, out: GpdOut, batch: DataBatch) -> tuple[Tensor, Tensor, Tensor]:
        target_s = batch.s.view(-1, 1).float()
        loss_adv = self._loss_adv_fn(out.s, target_s)
        target_y = batch.y.view(-1, 1).float()
        loss_clf = self._loss_clf_fn(out.y, target_y)
        return loss_adv, loss_clf, loss_adv + loss_clf

    def _inference_step(self, batch: DataBatch, stage: Stage) -> dict[str, Tensor]:
        model_out: GpdOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out, batch)
        logs = {
            f"{stage}/loss": loss.item(),
            f"{stage}/loss_adv": loss_adv.item(),
            f"{stage}/loss_clf": loss_clf.item(),
        }

        for _label in ("s", "y"):
            tm_acc = self.accs[f"{stage}_{_label}"]
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

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[LRScheduler]]:
        opt = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        sched = CosineAnnealingWarmRestarts(
            optimizer=opt, T_0=self.lr_initial_restart, T_mult=self.lr_restart_mult
        )
        return [opt], [sched]

    @implements(pl.LightningModule)
    def test_epoch_end(self, output_results: list[Mapping[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(output_results=output_results, stage="test")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> None:
        opt = self.optimizers()
        opt.zero_grad()

        model_out: GpdOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out, batch)

        logs = {
            f"train/adv_loss": loss_adv.item(),
            f"train/clf_loss": loss_clf.item(),
            f"train/loss": loss.item(),
        }
        compute_proj_grads(model=self.enc, loss_p=loss_clf, loss_a=loss_adv, alpha=1.0)
        compute_grad(model=self.adv, loss=loss_adv)
        compute_grad(model=self.clf, loss=loss_clf)

        for _label in ("s", "y"):
            tm_acc = self.accs[f"fit_{_label}"]
            _target = getattr(batch, _label).view(-1).long()
            _acc = tm_acc(getattr(model_out, _label).argmax(-1), _target)
            logs.update({f"train/acc_{_label}": _acc})
        self.log_dict(logs)
        opt.step()

        if self.lr_sched_interval == "step" and self.global_step % self.lr_sched_freq == 0:
            sch = self.lr_schedulers()
            sch.step()
        if self.lr_sched_interval == "epoch" and self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: list[Mapping[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(output_results=output_results, stage="validate")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="validate")

    @implements(nn.Module)
    def forward(self, x: Tensor) -> GpdOut:
        z = self.enc(x)
        y = self.clf(z)
        s = self.adv(z)
        return GpdOut(s=s, y=y)

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
