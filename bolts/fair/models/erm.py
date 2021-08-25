"""ERM Baseline Model."""
from __future__ import annotations

import ethicml as em
from kit import implements
from kit.decorators import parsable
from kit.torch import CrossEntropyLoss, ReductionType, TrainingMode
import pandas as pd
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import nn

from bolts.data.structures import TernarySample
from bolts.models import PBModel
from bolts.models.erm import ERMClassifier
from bolts.models.utils import aggregate_over_epoch, prediction
from bolts.types import Loss, Stage

__all__ = ["ERMClassifierF"]


class ERMClassifierF(ERMClassifier):
    """Empirical Risk Minimisation baseline."""

    @parsable
    def __init__(
        self,
        *,
        encoder: nn.Module,
        clf: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
        loss_fn: Loss = CrossEntropyLoss(reduction=ReductionType.mean),
    ) -> None:
        model = nn.Sequential(encoder, clf)
        super().__init__(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
            loss_fn=loss_fn,
        )
        self.encoder = encoder
        self.clf = clf

    @implements(ERMClassifier)
    @torch.no_grad()
    def inference_step(self, batch: TernarySample, *, stage: Stage) -> STEP_OUTPUT:
        results_dict = super().inference_step(batch=batch, stage=stage)
        results_dict["subgroups"] = batch.s
        return results_dict

    @implements(PBModel)
    @torch.no_grad()
    def inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> dict[str, float]:
        logits_all = aggregate_over_epoch(outputs=outputs, metric="logits")
        targets_all = aggregate_over_epoch(outputs=outputs, metric="targets")
        subgroups_all = aggregate_over_epoch(outputs=outputs, metric="subgroups")

        preds_all = prediction(logits_all)

        dt = em.DataTuple(
            x=pd.DataFrame(torch.rand_like(subgroups_all).detach().cpu().numpy(), columns=["x0"]),
            s=pd.DataFrame(subgroups_all.detach().cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(targets_all.detach().cpu().numpy(), columns=["y"]),
        )

        results_dict = em.run_metrics(
            predictions=em.Prediction(hard=pd.Series(preds_all.detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

        return results_dict
