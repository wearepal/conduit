"""LAFTR model."""
import itertools
from collections import namedtuple
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import ethicml as em
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from kit import implements
from torch import Tensor, nn, optim

__all__ = ["Laftr"]

from fair_bolts.datasets.ethicml_datasets import DataBatch

ModelOut = namedtuple("ModelOut", ["y", "z", "s", "x"])
FairnessType = Enum("FairnessType", "DP EO EqOp")


class Laftr(pl.LightningModule):
    """Learning Adversarially Fair and Transferrable Representations model."""

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        lr_gamma: float,
        disc_steps: int,
        fairness: str,
        recon_weight: float,
        clf_weight: float,
        adv_weight: float,
        enc: nn.Module,
        dec: nn.Module,
        adv: nn.Module,
        clf: nn.Module,
    ):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.adv = adv
        self.clf = clf

        self.laftr_params = itertools.chain(
            [*self.enc.parameters(), *self.dec.parameters(), *self.clf.parameters()]
        )
        self.adv_params = self.adv.parameters()

        self._clf_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self._recon_loss = nn.L1Loss(reduction="mean")
        self._adv_clf_loss = nn.L1Loss(reduction="mean")

        self.disc_steps = disc_steps
        self.fairness = FairnessType[fairness]
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.clf_weight = clf_weight
        self.adv_weight = adv_weight
        self.recon_weight = recon_weight

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self._target_name: str = "y"

    @property
    def target(self) -> str:
        return self._target_name

    @target.setter
    def target(self, target: str) -> None:
        self._target_name = target

    def _inference_epoch_end(self, output_results: List[Dict[str, Tensor]], stage: str) -> None:
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
            predictions=em.Prediction(hard=pd.Series(all_preds.detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

        tm_acc = self.val_acc if stage == "val" else self.test_acc
        acc = tm_acc.compute().item()
        results_dict = {f"{stage}/acc": acc}
        results_dict.update({f"{stage}/{self.target}_{k}": v for k, v in results.items()})

        self.log_dict(results_dict)

    def _inference_step(self, batch: DataBatch, stage: str) -> Dict[str, Tensor]:
        model_out = self(batch.x, batch.s)
        laftr_loss = self._loss_laftr(model_out.y, model_out.x, batch)
        adv_loss = self._loss_adv(model_out.s, batch)
        tm_acc = self.val_acc if stage == "val" else self.train_acc
        target = batch.y.long() if len(batch.y.shape) == 2 else batch.y.unsqueeze(-1).long()
        acc = tm_acc(model_out.y >= 0, target)
        self.log_dict(
            {
                f"{stage}/loss": (laftr_loss + adv_loss).item(),
                f"{stage}/model_loss": laftr_loss.item(),
                f"{stage}/adv_loss": adv_loss.item(),
                f"{stage}/{self.target}_acc": acc,
            }
        )
        return {"y": batch.y, "s": batch.s, "preds": model_out.y.sigmoid().round().squeeze(-1)}

    def _loss_adv(self, s_pred: Tensor, batch: DataBatch) -> Tensor:
        # For Demographic Parity, for EqOpp is a different loss term.
        s_target = batch.s.unsqueeze(-1) if len(batch.s.shape) == 1 else batch.s
        y_target = batch.y.unsqueeze(-1) if len(batch.y.shape) == 1 else batch.y
        if self.fairness is FairnessType.DP:
            s0 = self._adv_clf_loss(s_pred[batch.s == 0], s_target[batch.s == 0])
            s1 = self._adv_clf_loss(s_pred[batch.s == 1], s_target[batch.s == 1])
            loss = (s0 + s1) / 2
        elif self.fairness is FairnessType.EO:
            loss = torch.tensor(0.0).to(self.device)
            for s, y in itertools.product([0, 1], repeat=2):
                if len(batch.s[(batch.s == s) & (y_target == y)]) > 0:
                    loss += self._adv_clf_loss(
                        s_pred[(s_target == s) & (y_target == y)],
                        batch.s[(s_target == s) & (y_target == y)],
                    )
            loss = 2 - loss
        elif self.fairness is FairnessType.EqOp:
            # TODO: How to best handle this if no +ve samples in the batch?
            loss = torch.tensor(0.0).to(self.device)
            for s in (0, 1):
                if len(batch.s[(s_target == s) & (y_target == 1)]) > 0:
                    loss += self._adv_clf_loss(
                        s_pred[(s_target == s) & (y_target == 1)],
                        batch.s[(s_target == s) & (y_target == 1)],
                    )
            loss = 2 - loss
        else:
            raise RuntimeError("Only DP and EO fairness accepted.")
        self.log(f"{self.fairness}_adv_loss", self.adv_weight * loss)
        return self.adv_weight * loss

    def _loss_laftr(self, y_pred: Tensor, recon: Tensor, batch: DataBatch) -> Tensor:
        target = batch.y.float() if len(batch.y.shape) == 2 else batch.y.unsqueeze(-1).float()
        clf_loss = self._clf_loss(y_pred, target)
        recon_loss = self._recon_loss(recon, batch.x)
        self.log_dict(
            {"clf_loss": self.clf_weight * clf_loss, "recon_loss": self.recon_weight * recon_loss}
        )
        return self.clf_weight * clf_loss + self.recon_weight * recon_loss

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.ExponentialLR]]:
        opt_laftr = optim.AdamW(self.laftr_params, lr=self.lr, weight_decay=self.weight_decay)
        opt_adv = optim.AdamW(self.adv_params, lr=self.lr, weight_decay=self.weight_decay)

        sched_laftr = optim.lr_scheduler.ExponentialLR(optimizer=opt_laftr, gamma=self.lr_gamma)
        sched_adv = optim.lr_scheduler.ExponentialLR(optimizer=opt_adv, gamma=self.lr_gamma)

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
    def test_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        self._inference_epoch_end(output_results=output_results, stage="test")

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> Dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int, optimizer_idx: int) -> Tensor:
        if optimizer_idx == 0:
            # Main model update
            self.set_requires_grad(self.adv, requires_grad=False)
            model_out = self(batch.x, batch.s)
            laftr_loss = self._loss_laftr(model_out.y, model_out.x, batch)
            adv_loss = self._loss_adv(model_out.s, batch)
            target = batch.y.long() if len(batch.y.shape) == 2 else batch.y.unsqueeze(-1).long()
            acc = self.train_acc(model_out.y >= 0, target)
            self.log_dict(
                {
                    f"train/loss": (laftr_loss + adv_loss).item(),
                    f"train/model_loss": laftr_loss.item(),
                    f"train/acc": acc,
                }
            )
            return laftr_loss + adv_loss
        elif optimizer_idx == 1:
            # Adversarial update
            self.set_requires_grad([self.enc, self.dec, self.clf], requires_grad=False)
            self.set_requires_grad(self.adv, requires_grad=True)
            model_out = self(batch.x, batch.s)
            adv_loss = self._loss_adv(model_out.s, batch)
            laftr_loss = self._loss_laftr(model_out.y, model_out.x, batch)
            target = batch.y.long() if len(batch.y.shape) == 2 else batch.y.unsqueeze(-1).long()
            acc = self.train_acc(model_out.y >= 0, target)
            self.log_dict(
                {
                    f"train/loss": (laftr_loss + adv_loss).item(),
                    f"train/adv_loss": adv_loss.item(),
                    f"train/acc": acc,
                }
            )
            return -(laftr_loss + adv_loss)
        else:
            raise RuntimeError("There should only be 2 optimizers, but 3rd received.")

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        self._inference_epoch_end(output_results=output_results, stage="val")

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> Dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="val")

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> ModelOut:
        embedding = self.enc(x)
        y_pred = self.clf(embedding)
        s_pred = self.adv(embedding)
        recon = self.dec(embedding, s)
        return ModelOut(y=y_pred, z=embedding, x=recon, s=s_pred)

    @staticmethod
    def set_requires_grad(nets: Union[nn.Module, List[nn.Module]], requires_grad: bool) -> None:
        """Change if gradients are tracked."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
