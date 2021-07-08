"""DANN (Domain Adversarial Neural Network) model."""
from typing import Dict, List, NamedTuple, Optional, Tuple

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, autograd, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler
import torchmetrics
from torchmetrics import MetricCollection

from bolts.common import Stage
from bolts.fair.data.structures import DataBatch
from bolts.fair.losses import CrossEntropy

__all__ = ["Dann"]


class DannOut(NamedTuple):
    s: Tensor
    y: Tensor


class GradReverse(autograd.Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx: autograd.Function, x: Tensor, lambda_: float) -> Tensor:
        """Do GRL."""
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Do GRL."""
        return -ctx.lambda_ * grad_output, None


def grad_reverse(features: Tensor, lambda_: float = 1.0) -> Tensor:
    """Gradient Reversal layer."""
    return GradReverse.apply(features, lambda_)


class Dann(pl.LightningModule):
    """Ganin's Domain Adversarial NN."""

    def __init__(
        self,
        adv: nn.Module,
        enc: nn.Module,
        clf: nn.Module,
        lr: float,
        weight_decay: float,
        grl_lambda: float = 1.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: Literal["step", "epoch"] = "epoch",
        lr_sched_freq: int = 1,
    ) -> None:
        super().__init__()
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

        self._loss_adv_fn = CrossEntropy()
        self._loss_clf_fn = CrossEntropy()

        self.accs = MetricCollection(
            {
                f"{stage}_{label}": torchmetrics.Accuracy()
                for stage in ("train", "test", "val")
                for label in ("s", "y")
            }
        )

    def _inference_epoch_end(
        self, output_results: List[Dict[str, Tensor]], stage: Stage
    ) -> Dict[str, Tensor]:
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

    def _get_losses(self, out: DannOut, batch: DataBatch) -> Tuple[Tensor, Tensor, Tensor]:
        target_s = batch.s.view(-1, 1).float()
        loss_adv = self._loss_adv_fn(out.s, target_s)
        target_y = batch.y.view(-1, 1).float()
        loss_clf = self._loss_clf_fn(out.y, target_y)
        return loss_adv, loss_clf, loss_adv + loss_clf

    def _inference_step(self, batch: DataBatch, stage: Stage) -> Dict[str, Tensor]:
        model_out: DannOut = self.forward(batch.x)
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
    ) -> Tuple[List[optim.Optimizer], List[_LRScheduler]]:
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
    def test_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(output_results=output_results, stage="test")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> Dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        model_out: DannOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out, batch)

        logs = {
            f"train/adv_loss": loss_adv.item(),
            f"train/clf_loss": loss_clf.item(),
            f"train/loss": loss.item(),
        }
        for _label in ("s", "y"):
            tm_acc = self.accs[f"train_{_label}"]
            _target = getattr(batch, _label).view(-1).long()
            _acc = tm_acc(getattr(model_out, _label).argmax(-1), _target)
            logs.update({f"train/acc_{_label}": _acc})
        self.log_dict(logs)
        return loss

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(output_results=output_results, stage="val")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> Dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="val")

    @implements(nn.Module)
    def forward(self, x: Tensor) -> DannOut:
        z = self.enc(x)
        y = self.clf(z)
        s = self.adv(grad_reverse(z, lambda_=self.grl_lambda))
        return DannOut(s=s, y=y)

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
