"""LAFTR model."""
from __future__ import annotations
import itertools
from typing import Any, Mapping, NamedTuple

import ethicml as em
from kit import implements
from kit.torch import CrossEntropyLoss, ReductionType, TrainingMode
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchmetrics

from bolts.data.structures import TernarySample
from bolts.fair.misc import FairnessType
from bolts.models.base import ModelBase
from bolts.structures import LRScheduler, MetricDict, Stage

__all__ = ["Laftr"]


class ModelOut(NamedTuple):
    y: Tensor
    z: Tensor
    s: Tensor
    x: Tensor


class Laftr(ModelBase):
    """Learning Adversarially Fair and Transferrable Representations model.

    The model is only defined with respect to binary S and binary Y.
    """

    def __init__(
        self,
        *,
        lr: float,
        weight_decay: float,
        disc_steps: int,
        fairness: FairnessType,
        recon_weight: float,
        clf_weight: float,
        adv_weight: float,
        enc: nn.Module,
        dec: nn.Module,
        adv: nn.Module,
        clf: nn.Module,
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
        self.dec = dec
        self.adv = adv
        self.clf = clf

        self._clf_loss = CrossEntropyLoss(reduction=ReductionType.mean)
        self._recon_loss = nn.L1Loss(reduction="mean")
        self._adv_clf_loss = nn.L1Loss(reduction="none")

        self.disc_steps = disc_steps
        self.fairness = fairness

        self.clf_weight = clf_weight
        self.adv_weight = adv_weight
        self.recon_weight = recon_weight

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self._target_name: str = "y"

    @implements(ModelBase)
    def _inference_epoch_end(self, outputs: list[Mapping[str, Tensor]], stage: Stage) -> MetricDict:
        all_y = torch.cat([output_step["y"] for output_step in outputs], 0)
        all_s = torch.cat([output_step["s"] for output_step in outputs], 0)
        all_preds = torch.cat([output_step["preds"] for output_step in outputs], 0)

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
        results_dict = {f"{stage.value}/acc": tm_acc.compute()}
        results_dict.update(
            {f"{stage.value}/{self.target_name}_{k}": v for k, v in results.items()}
        )

        return results_dict

    @implements(ModelBase)
    def _inference_step(self, batch: TernarySample, *, stage: Stage) -> STEP_OUTPUT:
        model_out = self.forward(x=batch.x, s=batch.s)
        laftr_loss = self._loss_laftr(y_pred=model_out.y, recon=model_out.x, batch=batch)
        adv_loss = self._loss_adv(s_pred=model_out.s, batch=batch)
        tm_acc = self.val_acc if stage is Stage.validate else self.test_acc
        target = batch.y.view(-1).long()
        _acc = tm_acc(model_out.y.argmax(-1), target)
        self.log_dict(
            {
                f"{stage.value}/loss": (laftr_loss + adv_loss).item(),
                f"{stage.value}/model_loss": laftr_loss.item(),
                f"{stage.value}/adv_loss": adv_loss.item(),
                f"{stage.value}/{self.target_name}_acc": _acc,
            }
        )
        return {
            "y": batch.y.view(-1),
            "s": batch.s.view(-1),
            "preds": model_out.y.sigmoid().round().squeeze(-1),
        }

    def _loss_adv(self, s_pred: Tensor, *, batch: TernarySample) -> Tensor:
        # For Demographic Parity, for EqOpp is a different loss term.
        if self.fairness is FairnessType.DP:
            losses = self._adv_clf_loss(s_pred, batch.s.view(-1, 1))
            for s in (0, 1):
                mask = batch.s.view(-1) == s
                losses[mask] /= mask.sum()
            loss = 1 - losses.sum() / 2
        elif self.fairness is FairnessType.EO:
            unweighted_loss = self._adv_clf_loss(s_pred, batch.s.view(-1, 1))
            count = 0
            for s, y in itertools.product([0, 1], repeat=2):
                count += 1
                mask = (batch.s.view(-1) == s) & (batch.y.view(-1) == y)
                unweighted_loss[mask] /= mask.sum()
            loss = 2 - unweighted_loss.sum() / count
        elif self.fairness is FairnessType.EqOp:
            # TODO: How to best handle this if no +ve samples in the batch?
            unweighted_loss = self._adv_clf_loss(s_pred, batch.s.view(-1, 1))
            for s in (0, 1):
                mask = (batch.s.view(-1) == s) & (batch.y.view(-1) == 1)
                unweighted_loss[mask] /= mask.sum()
            unweighted_loss[batch.y.view(-1) == 0] *= 0.0
            loss = 2 - unweighted_loss.sum() / 2
        elif self.fairness is FairnessType.No:
            loss = s_pred.sum() * 0
        self.log(f"{self.fairness}_adv_loss", self.adv_weight * loss)
        return self.adv_weight * loss

    def _loss_laftr(self, y_pred: Tensor, *, recon: Tensor, batch: TernarySample) -> Tensor:
        clf_loss = self._clf_loss(y_pred, target=batch.y)
        recon_loss = self._recon_loss(recon, target=batch.x)
        self.log_dict(
            {"clf_loss": self.clf_weight * clf_loss, "recon_loss": self.recon_weight * recon_loss}
        )
        return self.clf_weight * clf_loss + self.recon_weight * recon_loss

    @implements(ModelBase)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[Mapping[str, LRScheduler | int | TrainingMode]]]:
        laftr_params = itertools.chain(
            [*self.enc.parameters(), *self.dec.parameters(), *self.clf.parameters()]
        )
        adv_params = self.adv.parameters()

        opt_laftr = optim.AdamW(laftr_params, lr=self.lr, weight_decay=self.weight_decay)
        opt_adv = optim.AdamW(adv_params, lr=self.lr, weight_decay=self.weight_decay)

        sched_laftr = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer=opt_laftr, T_0=self.lr_initial_restart, T_mult=self.lr_restart_mult
            ),
            "interval": self.lr_sched_interval.name,
            "frequency": self.lr_sched_freq,
        }
        sched_adv = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer=opt_adv, T_0=self.lr_initial_restart, T_mult=self.lr_restart_mult
            ),
            "interval": self.lr_sched_interval.name,
            "frequency": self.lr_sched_freq,
        }

        return [opt_laftr, opt_adv], [sched_laftr, sched_adv]

    @implements(pl.LightningModule)
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure: Any,
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool,
    ) -> None:
        # update main model every N steps
        if optimizer_idx == 0 and (batch_idx + 1) % self.disc_steps == 0:
            optimizer.step(closure=optimizer_closure)
        if optimizer_idx == 1:  # update discriminator opt every step
            optimizer.step(closure=optimizer_closure)

    @implements(pl.LightningModule)
    def training_step(self, batch: TernarySample, batch_idx: int, optimizer_idx: int) -> Tensor:
        if optimizer_idx == 0:
            # Main model update
            self.set_requires_grad(self.adv, requires_grad=False)
            model_out = self.forward(x=batch.x, s=batch.s)
            laftr_loss = self._loss_laftr(y_pred=model_out.y, recon=model_out.x, batch=batch)
            adv_loss = self._loss_adv(s_pred=model_out.s, batch=batch)
            target = batch.y.view(-1).long()
            _acc = self.train_acc(model_out.y.argmax(-1), target)
            self.log_dict(
                {
                    f"train/loss": (laftr_loss + adv_loss).item(),
                    f"train/model_loss": laftr_loss.item(),
                    f"train/acc": _acc,
                }
            )
            return laftr_loss + adv_loss
        elif optimizer_idx == 1:
            # Adversarial update
            self.set_requires_grad([self.enc, self.dec, self.clf], requires_grad=False)
            self.set_requires_grad(self.adv, requires_grad=True)
            model_out = self.forward(x=batch.x, s=batch.s)
            adv_loss = self._loss_adv(s_pred=model_out.s, batch=batch)
            laftr_loss = self._loss_laftr(y_pred=model_out.y, recon=model_out.x, batch=batch)
            target = batch.y.view(-1).long()
            _acc = self.train_acc(model_out.y.argmax(-1), target)
            self.log_dict(
                {
                    f"train/loss": (laftr_loss + adv_loss).item(),
                    f"train/adv_loss": adv_loss.item(),
                    f"train/acc": _acc,
                }
            )
            return -(laftr_loss + adv_loss)
        else:
            raise RuntimeError("There should only be 2 optimizers, but 3rd received.")

    @staticmethod
    def set_requires_grad(nets: nn.Module | list[nn.Module], requires_grad: bool) -> None:
        """Change if gradients are tracked."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    @implements(nn.Module)
    def forward(self, x: Tensor, *, s: Tensor) -> ModelOut:
        embedding = self.enc(x)
        y_pred = self.clf(embedding)
        s_pred = self.adv(embedding)
        recon = self.dec(embedding, s)
        return ModelOut(y=y_pred, z=embedding, x=recon, s=s_pred)
