"""ERM Baseline Model."""
from typing import Dict, List, Tuple

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import _LRScheduler
import torchmetrics
from typing_extensions import Literal

from bolts.datasets.ethicml_datasets import DataBatch

Stage = Literal["train", "val", "test"]


class ErmBaseline(pl.LightningModule):
    """Empirical Risk Minimisation baseline."""

    def __init__(
        self,
        enc: nn.Module,
        clf: nn.Module,
        lr: float,
        weight_decay: float,
        lr_gamma: float,
    ) -> None:
        super().__init__()
        self.learning_rate = lr
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.enc = enc
        self.clf = clf
        self.net = nn.Sequential(self.enc, self.clf)
        self._loss_fn = nn.BCEWithLogitsLoss()

        self._target_name = "y"

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.batch_norm = True  # TODO: fixme

    @property
    def target(self) -> str:
        return self._target_name

    @target.setter
    def target(self, target: str) -> None:
        self._target_name = target

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

        tm_acc = self.val_acc if stage == "val" else self.test_acc
        acc = tm_acc.compute().item()
        results_dict = {f"{stage}/acc": acc}
        results_dict.update({f"{stage}/{self.target}_{k}": v for k, v in results.items()})
        return results_dict

    def _inference_step(self, batch: DataBatch, stage: Stage) -> Dict[str, Tensor]:
        logits = self.forward(batch.x)
        loss = self._get_loss(logits, batch)
        tm_acc = self.val_acc if stage == "val" else self.test_acc
        target = batch.y.view(-1, 1).long()
        acc = tm_acc(logits >= 0, target)
        self.log_dict(
            {
                f"{stage}/loss": loss.item(),
                f"{stage}/{self.target}_acc": acc,
            }
        )
        return {"y": batch.y, "s": batch.s, "preds": logits.sigmoid().round().squeeze(-1)}

    def _get_loss(self, logits: Tensor, batch: DataBatch) -> Tensor:
        target = batch.y.view(-1, 1).float()
        return self._loss_fn(input=logits, target=target)

    def reset_parameters(self) -> None:
        """Reset the models."""
        self.enc.apply(self._maybe_reset_parameters)
        self.clf.apply(self._maybe_reset_parameters)

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Tuple[List[optim.Optimizer], List[_LRScheduler]]:
        opt = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=self.lr_gamma)
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
        logits = self.forward(batch.x)
        loss = self._get_loss(logits, batch)
        target = batch.y.view(-1, 1).long()
        acc = self.train_acc(logits >= 0, target)
        self.log_dict(
            {
                f"train/loss": loss.item(),
                f"train/acc": acc,
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
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
