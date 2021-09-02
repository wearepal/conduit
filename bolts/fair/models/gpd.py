"""Zhang Gradient Projection Debiasing Baseline Model."""
from __future__ import annotations
from typing import NamedTuple, cast

import ethicml as em
from kit import implements
from kit.torch import CrossEntropyLoss, TrainingMode
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from bolts.data.structures import TernarySample
from bolts.models.base import PBModel
from bolts.models.utils import aggregate_over_epoch, prediction, prefix_keys
from bolts.types import LRScheduler, Stage

__all__ = ["GPD"]


def compute_proj_grads(*, model: nn.Module, loss_p: Tensor, loss_a: Tensor, alpha: float) -> None:
    """Computes the adversarial-gradient projection term.

    Args:
        model: Model whose parameters the gradients are to be computed w.r.t.
        loss_p: Prediction loss.
        loss_a: Adversarial loss.
        alpha: Pre-factor for adversarial loss.
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


class ModelOut(NamedTuple):
    s: Tensor
    y: Tensor


class GPD(PBModel):
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

        self.automatic_optimization = False  # Mark for manual optimization

    @implements(PBModel)
    @torch.no_grad()
    def inference_step(self, batch: TernarySample, *, stage: Stage) -> dict[str, Tensor]:
        assert isinstance(batch.x, Tensor)
        model_out = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out=model_out, batch=batch)
        logging_dict = {
            f"loss": loss.item(),
            f"loss_adv": loss_adv.item(),
            f"loss_clf": loss_clf.item(),
        }
        logging_dict = prefix_keys(dict_=logging_dict, prefix=str(stage), sep="/")
        self.log_dict(logging_dict)

        return {
            "targets": batch.y.view(-1),
            "subgroup_inf": batch.s.view(-1),
            "logits_y": model_out.y,
        }

    @implements(PBModel)
    def inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> dict[str, float]:
        targets_all = aggregate_over_epoch(outputs=outputs, metric="targets")
        subgroup_inf_all = aggregate_over_epoch(outputs=outputs, metric="subgroup_inf")
        logits_y_all = aggregate_over_epoch(outputs=outputs, metric="logits_y")

        preds_y_all = prediction(logits_y_all)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(subgroup_inf_all).detach().cpu().numpy(),
                columns=["x0"],
            ),
            s=pd.DataFrame(subgroup_inf_all.detach().cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(targets_all.detach().cpu().numpy(), columns=["y"]),
        )

        return em.run_metrics(
            predictions=em.Prediction(hard=pd.Series(preds_y_all.detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

    def _get_losses(
        self, model_out: ModelOut, *, batch: TernarySample
    ) -> tuple[Tensor, Tensor, Tensor]:
        loss_adv = self._loss_adv_fn(model_out.s, target=batch.s)
        loss_clf = self._loss_clf_fn(model_out.y, target=batch.y)
        return loss_adv, loss_clf, loss_adv + loss_clf

    @implements(pl.LightningModule)
    def training_step(self, batch: TernarySample, batch_idx: int) -> None:
        assert isinstance(batch.x, Tensor)
        opt = cast(Optimizer, self.optimizers())

        opt.zero_grad()

        model_out: ModelOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out=model_out, batch=batch)

        logging_dict = {
            f"adv_loss": loss_adv.item(),
            f"clf_loss": loss_clf.item(),
            f"loss": loss.item(),
        }
        logging_dict = prefix_keys(dict_=logging_dict, prefix="train", sep="/")
        self.log_dict(logging_dict)

        compute_proj_grads(model=self.enc, loss_p=loss_clf, loss_a=loss_adv, alpha=1.0)
        compute_grad(model=self.adv, loss=loss_adv)
        compute_grad(model=self.clf, loss=loss_clf)

        opt.step()

        if (self.lr_sched_interval is TrainingMode.step) and (
            self.global_step % self.lr_sched_freq == 0
        ):
            sch = cast(LRScheduler, self.lr_schedulers())
            sch.step()
        if (self.lr_sched_interval is TrainingMode.epoch) and self.trainer.is_last_batch:
            sch = cast(LRScheduler, self.lr_schedulers())
            sch.step()

    @implements(nn.Module)
    def forward(self, x: Tensor) -> ModelOut:
        embedding = self.enc(x)
        y_pred = self.clf(embedding)
        s_pred = self.adv(embedding)
        return ModelOut(y=y_pred, s=s_pred)
