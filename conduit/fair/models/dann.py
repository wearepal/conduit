"""DANN (Domain Adversarial Neural Network) model."""
from typing import Dict, NamedTuple, Optional, Tuple

import ethicml as em
from kit import implements
from kit.torch import CrossEntropyLoss, TrainingMode
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import Tensor, autograd, nn

from conduit.data.structures import TernarySample
from conduit.models.base import CdtModel
from conduit.models.utils import aggregate_over_epoch, prefix_keys
from conduit.types import Stage

__all__ = ["DANN"]


class ModelOut(NamedTuple):
    s: Tensor
    y: Tensor


class GradReverse(autograd.Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx: autograd.Function, x: Tensor, lambda_: float) -> Tensor:
        """Do GRL."""
        if lambda_ < 0:
            raise ValueError("Argument 'lambda_' to GradReverse.forward must be non-negative.")
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Reverse (and optionally scale) the gradient."""
        return -ctx.lambda_ * grad_output, None


def grad_reverse(features: Tensor, *, lambda_: float = 1.0) -> Tensor:
    """Gradient Reversal layer."""
    return GradReverse.apply(features, lambda_)


class DANN(CdtModel):
    """Ganin's Domain Adversarial NN."""

    def __init__(
        self,
        *,
        adv: nn.Module,
        enc: nn.Module,
        clf: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        grl_lambda: float = 1.0,
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

        self._loss_adv_fn = CrossEntropyLoss()
        self._loss_clf_fn = CrossEntropyLoss()

    def _get_losses(
        self, model_out: ModelOut, *, batch: TernarySample
    ) -> Tuple[Tensor, Tensor, Tensor]:
        target_s = batch.s.view(-1, 1).float()
        loss_adv = self._loss_adv_fn(model_out.s, target=target_s)
        target_y = batch.y.view(-1, 1).float()
        loss_clf = self._loss_clf_fn(model_out.y, target=target_y)
        return loss_adv, loss_clf, loss_adv + loss_clf

    @implements(pl.LightningModule)
    def training_step(self, batch: TernarySample, batch_idx: int) -> Tensor:
        assert isinstance(batch.x, Tensor)
        model_out: ModelOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out=model_out, batch=batch)

        logging_dict = {
            f"{Stage.fit}/adv_loss": loss_adv.item(),
            f"{Stage.fit}": loss_clf.item(),
            f"{Stage.fit}/loss": loss.item(),
        }
        logging_dict = prefix_keys(dict_=logging_dict, prefix="train", sep="/")
        self.log_dict(logging_dict)

        return loss

    @implements(CdtModel)
    def inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> Dict[str, float]:
        logits_all = aggregate_over_epoch(outputs=outputs, metric="logits")
        targets_all = aggregate_over_epoch(outputs=outputs, metric="targets")
        subgroup_inf_all = aggregate_over_epoch(outputs=outputs, metric="subgroup_inf")

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(subgroup_inf_all, dtype=torch.float).detach().cpu().numpy(),
                columns=["x0"],
            ),
            s=pd.DataFrame(subgroup_inf_all.detach().cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(targets_all.detach().cpu().numpy(), columns=["y"]),
        )

        return em.run_metrics(
            predictions=em.Prediction(hard=pd.Series(logits_all.argmax(-1).detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

    @implements(CdtModel)
    def inference_step(self, batch: TernarySample, *, stage: Stage) -> STEP_OUTPUT:
        assert isinstance(batch.x, Tensor)
        model_out: ModelOut = self.forward(batch.x)
        loss_adv, loss_clf, loss = self._get_losses(model_out=model_out, batch=batch)
        logging_dict = {
            "loss": loss.item(),
            "loss_adv": loss_adv.item(),
            "loss_clf": loss_clf.item(),
        }
        logging_dict = prefix_keys(dict_=logging_dict, prefix=str(stage), sep="/")
        self.log_dict(logging_dict)

        return {
            "targets": batch.y.view(-1),
            "subgroup_inf": batch.s.view(-1),
            "logits": model_out.y,
        }

    @torch.no_grad()
    def _maybe_reset_parameters(self, module: nn.Module) -> None:
        if (module != self) and hasattr(module, 'reset_parameters'):
            module.reset_parameters()  # type: ignore

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset the models."""
        self.apply(self._maybe_reset_parameters)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> ModelOut:
        z = self.enc(x)
        y = self.clf(z)
        s = self.adv(grad_reverse(z, lambda_=self.grl_lambda))
        return ModelOut(s=s, y=y)
