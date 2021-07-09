"""Implementation of ICLR 21 Fair Mixup."""
from __future__ import annotations
from typing import Callable, Mapping, Optional

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor, nn, optim
from torch.distributions import Beta
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchmetrics

from bolts.common import Stage
from bolts.fair.data import DataBatch
from bolts.fair.losses import CrossEntropy
from bolts.fair.models import LRScheduler, SchedInterval

__all__ = ["FairMixup"]


class FairMixup(pl.LightningModule):
    def __init__(
        self,
        enc: nn.Module,
        clf: nn.Module,
        lr: float,
        weight_decay: float,
        mixup_lambda: float | None = None,
        alpha: float = 1.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: SchedInterval = "epoch",
        lr_sched_freq: int = 1,
    ):
        super().__init__()
        self.enc = enc
        self.clf = clf
        self.net = nn.Sequential(self.enc, self.clf)
        self._loss_fn = CrossEntropy()

        self.mixup_lambda = mixup_lambda
        self.alpha = alpha
        self._target_name = "y"

        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.lr_initial_restart = lr_initial_restart
        self.lr_restart_mult = lr_restart_mult
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    @property
    def target(self) -> str:
        return self._target_name

    @target.setter
    def target(self, target: str) -> None:
        self._target_name = target

    def training_step(self, batch: DataBatch, batch_idx: int) -> STEP_OUTPUT:
        inputs, x_a, x_b, targets_a, targets_b, _, _, lam, mixup_stats = self.mixup_data(
            batch=batch, device=self.device, mix_lambda=self.mixup_lambda, alpha=self.alpha
        )

        logits = self.net(inputs)
        loss_sup = self.mixup_criterion(
            criterion=self._loss_fn, pred=logits, tgt_a=targets_a, tgt_b=targets_b, lam=lam
        )

        target = batch.y.view(-1).long()
        self.train_acc(logits.argmax(-1), target)
        self.log_dict(
            {
                f"train/loss": loss_sup.item(),
                f"train/acc": self.train_acc,
            }
        )

        ops = logits.sum()

        # Smoothness Regularization
        gradx = torch.autograd.grad(ops, inputs, create_graph=True)[0].view(inputs.shape[0], -1)
        x_d = (x_b - x_a).view(inputs.size(0), -1)
        grad_inn = (gradx * x_d).sum(1).view(-1)
        loss_grad = torch.abs(grad_inn.mean())

        loss = loss_sup + lam * loss_grad

        self.log_dict({"train/loss_sup": loss_sup.item(), "train/loss_mixup": loss_grad.item()})

        return loss

    def _get_loss(self, logits: Tensor, batch: DataBatch) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y)

    def _inference_step(self, batch: DataBatch, stage: Stage) -> dict[str, Tensor]:
        logits = self.net(batch.x)

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
            "preds": logits.softmax(-1)[:, 1],
        }

    def _inference_epoch_end(
        self, results: list[Mapping[str, Tensor]], stage: Stage
    ) -> dict[str, float]:
        all_y = torch.cat([_r["y"] for _r in results], 0)
        all_s = torch.cat([_r["s"] for _r in results], 0)
        all_preds = torch.cat([_r["preds"] for _r in results], 0)

        mean_preds = all_preds.mean(-1)
        mean_preds_s0 = all_preds[all_s == 0].mean(-1)
        mean_preds_s1 = all_preds[all_s == 1].mean(-1)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(all_s, dtype=float).detach().cpu().numpy(), columns=["x0"]
            ),
            s=pd.DataFrame(all_s.detach().cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(all_y.detach().cpu().numpy(), columns=["y"]),
        )

        results = em.run_metrics(
            predictions=em.Prediction(hard=pd.Series((all_preds > 0).detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

        tm_acc = self.val_acc if stage == "validate" else self.test_acc
        results_dict = {f"{stage}/acc": tm_acc.compute()}
        results_dict.update({f"{stage}/{self.target}_{k}": v for k, v in results.items()})
        results_dict.update(
            {
                f"{stage}/DP_Gap": abs(mean_preds_s0 - mean_preds_s1),
                f"{stage}/mean_pred": mean_preds,
            }
        )
        return results_dict

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: list[Mapping[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(results=output_results, stage="validate")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="validate")

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @implements(pl.LightningModule)
    def test_epoch_end(self, output_results: list[Mapping[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(results=output_results, stage="test")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="test")

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

    @staticmethod
    def mixup_data(
        *,
        batch: DataBatch,
        device: torch.device,
        mix_lambda: Optional[float],
        alpha: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, float]]:
        '''Returns mixed inputs, pairs of targets, and lambda'''
        lam = (
            Beta(
                torch.tensor([alpha]).to(device),
                torch.tensor([alpha]).to(device)
                # Potentially change alpha from a=1.0 to account for class imbalance?
            ).sample()
            if mix_lambda is None
            else torch.tensor([mix_lambda]).to(device)
        )

        batch_s0 = batch.x[batch.s.view(-1) == 0].to(device)
        batch_s1 = batch.x[batch.s.view(-1) == 1].to(device)

        batch_y_s0 = batch.y[batch.s.view(-1) == 0].to(device)
        batch_y_s1 = batch.y[batch.s.view(-1) == 1].to(device)

        torch.randperm(batch_s0.size()[0]).to(device)
        torch.randperm(batch_s1.size()[0]).to(device)

        xal = []
        xbl = []
        sal = []
        sbl = []
        yal = []
        ybl = []

        for x_a, s_a, y_a in zip(batch.x, batch.s, batch.y):
            if s_a == 0:
                xal.append(x_a)
                sal.append(s_a.unsqueeze(-1).float())
                yal.append(y_a.unsqueeze(-1).float())
                idx = torch.randint(batch_s1.shape[0], (1,))
                x_b = batch_s1[idx, :].squeeze(0)
                xbl.append(x_b)
                sbl.append(torch.ones_like(s_a).unsqueeze(-1).float())
                y_b = batch_y_s1[idx].float()
                ybl.append(y_b)

            elif s_a == 1:
                xal.append(x_a)
                sal.append(s_a.unsqueeze(-1).float())
                yal.append(y_a.unsqueeze(-1).float())
                idx = torch.randint(batch_s0.size()[0], (1,))
                x_b = batch_s0[idx, :].squeeze(0)
                xbl.append(x_b)
                sbl.append(torch.zeros_like(s_a).unsqueeze(-1).float())
                y_b = batch_y_s0[idx].float()
                ybl.append(y_b)

        x_a = torch.stack(xal, dim=0).to(device)
        x_b = torch.stack(xbl, dim=0).to(device)

        s_a = torch.stack(sal, dim=0).to(device)
        s_b = torch.stack(sbl, dim=0).to(device)

        y_a = torch.stack(yal, dim=0).to(device)
        y_b = torch.stack(ybl, dim=0).to(device)

        mix_stats = {
            "batch_stats/S0=sS0": sum(a == b for a, b in zip(s_a, s_b)) / batch.s.shape[0],
            "batch_stats/S0!=sS0": sum(a != b for a, b in zip(s_a, s_b)) / batch.s.shape[0],
            "batch_stats/all_s0": sum(a + b == 0 for a, b in zip(s_a, s_b)) / batch.s.shape[0],
            "batch_stats/all_s1": sum(a + b == 2 for a, b in zip(s_a, s_b)) / batch.s.shape[0],
        }

        mixed_x = lam * x_a + (1 - lam) * x_b
        return mixed_x.requires_grad_(True), x_a, x_b, s_a, s_b, y_a, y_b, lam, mix_stats

    @staticmethod
    def mixup_criterion(
        *,
        criterion: Callable[[Tensor, Tensor], Tensor],
        pred: Tensor,
        tgt_a: Tensor,
        tgt_b: Tensor,
        lam: Tensor,
    ) -> Tensor:
        return lam * criterion(pred, tgt_a) + (1 - lam) * criterion(pred, tgt_b)
