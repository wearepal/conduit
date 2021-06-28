"""ERM Baseline Model."""
from typing import Dict, List, Optional, Tuple

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, autograd, nn, optim
import torchmetrics
from typing_extensions import Literal

from bolts.datasets.ethicml_datasets import DataBatch

__all__ = ["Dann"]

Stage = Literal["train", "val", "test"]


class GradReverse(autograd.Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx: autograd.Function, x: Tensor, lambda_: float) -> Tensor:
        """Do GRL."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Do GRL."""
        return grad_output.neg().mul(ctx.lambda_), None


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
    ) -> None:
        super().__init__()
        self.grl_lambda = grl_lambda
        self.learning_rate = lr
        self.weight_decay = weight_decay

        self.adv = adv
        self.enc = enc
        self.clf = clf

        self._loss_adv_fn = nn.BCEWithLogitsLoss()
        self._loss_clf_fn = nn.BCEWithLogitsLoss()

        self.test_acc_s = torchmetrics.Accuracy()
        self.test_acc_y = torchmetrics.Accuracy()
        self.train_acc_s = torchmetrics.Accuracy()
        self.train_acc_y = torchmetrics.Accuracy()
        self.val_acc_s = torchmetrics.Accuracy()
        self.val_acc_y = torchmetrics.Accuracy()

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
            predictions=em.Prediction(hard=pd.Series(all_preds.detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

        tm_acc_s = self.val_acc_s if stage == "val" else self.test_acc_s
        tm_acc_y = self.val_acc_y if stage == "val" else self.test_acc_y
        acc_s = tm_acc_s.compute().item()
        acc_y = tm_acc_y.compute().item()
        results_dict = {f"{stage}/acc_s": acc_s, f"{stage}/acc_y": acc_y}
        results_dict.update({f"{stage}/{k}": v for k, v in results.items()})
        return results_dict

    def _inference_step(self, batch: DataBatch, stage: Stage) -> Dict[str, Tensor]:
        _s, _y = self.forward(batch.x)
        target_s = batch.s.view(-1, 1).float()
        loss_adv = self._loss_adv_fn(_s, target_s)
        target_y = batch.y.view(-1, 1).float()
        loss_clf = self._loss_clf_fn(_y, target_y)
        loss = loss_adv + loss_clf
        tm_acc_s = self.val_acc_s if stage == "val" else self.test_acc_s
        tm_acc_y = self.val_acc_y if stage == "val" else self.test_acc_y

        target_y = batch.y.view(-1, 1).long()
        y_acc = tm_acc_y(_y >= 0, target_y)

        target_s = batch.s.view(-1, 1).long()
        s_acc = tm_acc_s(_s >= 0, target_s)
        self.log_dict(
            {
                f"{stage}/loss": loss.item(),
                f"{stage}/loss_adv": loss_adv.item(),
                f"{stage}/loss_clf": loss_clf.item(),
                f"{stage}/acc_s": s_acc,
                f"{stage}/acc_y": y_acc,
            }
        )
        return {"y": batch.y, "s": batch.s, "preds": _y.sigmoid().round().squeeze(-1)}

    def reset_parameters(self) -> None:
        """Reset the models."""
        self.adv.apply(self._maybe_reset_parameters)
        self.enc.apply(self._maybe_reset_parameters)
        self.clf.apply(self._maybe_reset_parameters)

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> optim.Optimizer:
        return optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @implements(pl.LightningModule)
    def test_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(output_results=output_results, stage="test")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> Dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        _s, _y = self.forward(batch.x)
        target_s = batch.s.view(-1, 1).float()
        loss_adv = self._loss_adv_fn(_s, target_s)
        target_y = batch.y.view(-1, 1).float()
        loss_clf = self._loss_clf_fn(_y, target_y)
        loss = loss_adv + loss_clf

        target_s = batch.s.view(-1, 1).long()
        acc_s = self.train_acc_s(_s >= 0, target_s)
        target_y = batch.y.view(-1, 1).long()
        acc_y = self.train_acc_y(_y >= 0, target_y)
        self.log_dict(
            {
                f"train/adv_loss": loss_adv.item(),
                f"train/clf_loss": loss_clf.item(),
                f"train/loss": loss.item(),
                f"train/acc_s": acc_s,
                f"train/acc_y": acc_y,
            }
        )
        return loss

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        results_dict = self._inference_epoch_end(output_results=output_results, stage="val")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> Dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="val")

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.enc(x)
        y = self.clf(z)
        s = self.adv(grad_reverse(z, lambda_=self.grl_lambda))
        return s, y

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
