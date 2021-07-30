"""Implementation of ICLR 21 Fair Mixup.

https://github.com/chingyaoc/fair-mixup
"""
from __future__ import annotations
from typing import Callable, Dict, Mapping, NamedTuple, Optional

import ethicml as em
from kit import implements, parsable
from kit.torch import CrossEntropyLoss, ReductionType, TrainingMode
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor, nn
from torch.distributions import Beta
import torchmetrics

from bolts.common import MetricDict, Stage
from bolts.data import TernarySample
from bolts.fair.misc import FairnessType
from bolts.models.base import ModelBase

__all__ = ["FairMixup"]


class Mixed(NamedTuple):
    x: Tensor
    xa: Tensor
    xb: Tensor
    sa: Tensor
    sb: Tensor
    ya: Tensor
    yb: Tensor
    lam: Tensor
    stats: Dict[str, float]


class FairMixup(ModelBase):
    @parsable
    def __init__(
        self,
        enc: nn.Module,
        clf: nn.Module,
        lr: float,
        weight_decay: float,
        fairness: FairnessType,
        mixup_lambda: Optional[float] = None,
        alpha: float = 1.0,
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
        self.fairness = fairness
        self.mixup_lambda = mixup_lambda
        self.alpha = alpha

        self._target_name = "y"
        self._loss_fn = CrossEntropyLoss(reduction=ReductionType.mean)

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def _get_loss(self, logits: Tensor, batch: TernarySample) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y)

    @implements(pl.LightningModule)
    def training_step(self, batch: TernarySample, batch_idx: int) -> STEP_OUTPUT:
        mixed = self.mixup_data(
            batch=batch,
            device=self.device,
            mix_lambda=self.mixup_lambda,
            alpha=self.alpha,
            fairness=self.fairness,
        )

        logits = self.net(mixed.x)
        loss_sup = self.mixup_criterion(
            criterion=self._loss_fn, pred=logits, tgt_a=mixed.ya, tgt_b=mixed.yb, lam=mixed.lam
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
        gradx = torch.autograd.grad(ops, mixed.x, create_graph=True)[0].view(mixed.x.size(0), -1)
        x_d = (mixed.xb - mixed.xa).view(mixed.x.size(0), -1)
        grad_inn = (gradx * x_d).sum(1).flatten()
        loss_grad = torch.abs(grad_inn.mean())

        loss = loss_sup + mixed.lam * loss_grad

        self.log_dict({"train/loss_sup": loss_sup.item(), "train/loss_mixup": loss_grad.item()})

        return loss

    @implements(ModelBase)
    def _inference_step(self, batch: TernarySample, stage: Stage) -> STEP_OUTPUT:
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

    @implements(ModelBase)
    def _inference_epoch_end(
        self, output_results: list[Mapping[str, Tensor]], stage: Stage
    ) -> MetricDict:
        all_y = torch.cat([_r["y"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_preds = torch.cat([_r["preds"] for _r in output_results], 0)

        mean_preds = all_preds.mean(-1)
        mean_preds_s0 = all_preds[all_s == 0].mean(-1)
        mean_preds_s1 = all_preds[all_s == 1].mean(-1)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(all_s, dtype=torch.float).detach().cpu().numpy(), columns=["x0"]
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
        results_dict = {f"{stage}/acc": tm_acc.compute().item()}
        results_dict.update({f"{stage}/{self.target}_{k}": v for k, v in results.items()})
        results_dict.update(
            {
                f"{stage}/DP_Gap": abs(mean_preds_s0 - mean_preds_s1),
                f"{stage}/mean_pred": mean_preds,
            }
        )
        return results_dict

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @staticmethod
    def mixup_data(
        *,
        batch: TernarySample,
        device: torch.device,
        mix_lambda: Optional[float],
        alpha: float,
        fairness: FairnessType,
    ) -> Mixed:
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

        batches = {
            "x_s0": batch.x[batch.s.view(-1) == 0].to(device),
            "x_s1": batch.x[batch.s.view(-1) == 1].to(device),
            "y_s0": batch.y[batch.s.view(-1) == 0].to(device),
            "y_s1": batch.y[batch.s.view(-1) == 1].to(device),
            "x_s0_y0": batch.x[(batch.s.view(-1) == 0) & (batch.y.view(-1) == 0)].to(device),
            "x_s1_y0": batch.x[(batch.s.view(-1) == 1) & (batch.y.view(-1) == 0)].to(device),
            "x_s0_y1": batch.x[(batch.s.view(-1) == 0) & (batch.y.view(-1) == 1)].to(device),
            "x_s1_y1": batch.x[(batch.s.view(-1) == 1) & (batch.y.view(-1) == 1)].to(device),
            "s_s0_y0": batch.s[(batch.s.view(-1) == 0) & (batch.y.view(-1) == 0)].to(device),
            "s_s1_y0": batch.s[(batch.s.view(-1) == 1) & (batch.y.view(-1) == 0)].to(device),
            "s_s0_y1": batch.s[(batch.s.view(-1) == 0) & (batch.y.view(-1) == 1)].to(device),
            "s_s1_y1": batch.s[(batch.s.view(-1) == 1) & (batch.y.view(-1) == 1)].to(device),
        }
        xal = []
        xbl = []
        sal = []
        sbl = []
        yal = []
        ybl = []

        for x_a, s_a, y_a in zip(batch.x, batch.s, batch.y):
            xal.append(x_a)
            sal.append(s_a.unsqueeze(-1).float())
            yal.append(y_a.unsqueeze(-1).float())
            if (fairness is FairnessType.EqOp and y_a == 0) or fairness is FairnessType.No:
                xbl.append(x_a)
                sbl.append(s_a.unsqueeze(-1))
                ybl.append(y_a.unsqueeze(-1))
            elif fairness is FairnessType.EqOp:
                idx = torch.randint(batches[f"x_s{1 - int(s_a)}_y1"].size(0), (1,))
                x_b = batches[f"x_s{1 - int(s_a)}_y1"][idx, :].squeeze(0)
                xbl.append(x_b)
                sbl.append((torch.ones_like(s_a) * (1 - s_a)).unsqueeze(-1).float())
                y_b = torch.ones_like(y_a).unsqueeze(-1)
                ybl.append(y_b)
            elif fairness is FairnessType.DP:
                idx = torch.randint(batches[f"x_s{1-int(s_a)}"].size(0), (1,))
                x_b = batches[f"x_s{1-int(s_a)}"][idx, :].squeeze(0)
                xbl.append(x_b)
                sbl.append((torch.ones_like(s_a) * (1 - s_a)).unsqueeze(-1).float())
                y_b = batches[f"y_s{1-int(s_a)}"][idx].float()
                ybl.append(y_b)
            elif fairness is FairnessType.EO:
                idx = torch.randint(batches[f"x_s{1-int(s_a)}_y{int(y_a)}"].size(0), (1,))
                x_b = batches[f"x_s{1-int(s_a)}_y{int(y_a)}"][idx, :].squeeze(0)
                xbl.append(x_b)
                sbl.append((torch.ones_like(s_a) * (1 - s_a)).unsqueeze(-1).float())
                y_b = (torch.ones_like(y_a) * y_a).unsqueeze(-1)
                ybl.append(y_b)
        x_a = torch.stack(xal, dim=0).to(device)
        x_b = torch.stack(xbl, dim=0).to(device)

        s_a = torch.stack(sal, dim=0).to(device)
        s_b = torch.stack(sbl, dim=0).to(device)

        y_a = torch.stack(yal, dim=0).to(device)
        y_b = torch.stack(ybl, dim=0).to(device)

        mix_stats = {
            "batch_stats/S0=sS0": sum(a == b for a, b in zip(s_a, s_b)) / batch.s.size(0),
            "batch_stats/S0!=sS0": sum(a != b for a, b in zip(s_a, s_b)) / batch.s.size(0),
            "batch_stats/all_s0": sum(a + b == 0 for a, b in zip(s_a, s_b)) / batch.s.size(0),
            "batch_stats/all_s1": sum(a + b == 2 for a, b in zip(s_a, s_b)) / batch.s.size(0),
        }

        mixed_x = lam * x_a + (1 - lam) * x_b
        return Mixed(
            x=mixed_x.requires_grad_(True),
            xa=x_a,
            xb=x_b,
            sa=s_a,
            sb=s_b,
            ya=y_a,
            yb=y_b,
            lam=lam,
            stats=mix_stats,
        )

    @staticmethod
    def mixup_criterion(
        *,
        criterion: Callable[[Tensor, Tensor], Tensor],
        pred: Tensor,
        tgt_a: Tensor,
        tgt_b: Tensor,
        lam: Tensor,
    ) -> Tensor:
        return lam * criterion(input=pred, target=tgt_a) + (1 - lam) * criterion(
            input=pred, target=tgt_b
        )
