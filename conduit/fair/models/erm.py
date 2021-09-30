"""ERM Baseline Model."""
from typing import Dict, Optional

import ethicml as em
import pandas as pd
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from ranzen import implements
from ranzen.decorators import parsable
from ranzen.torch import TrainingMode
import torch
from torch import nn

from conduit.data.structures import TernarySample
from conduit.models import CdtModel
from conduit.models.erm import ERMClassifier
from conduit.models.utils import aggregate_over_epoch, prediction
from conduit.types import Loss, Stage

__all__ = ["ERMClassifierF"]


class ERMClassifierF(ERMClassifier):
    """Empirical Risk Minimisation baseline."""

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
        loss_fn: Optional[Loss] = None,
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
        results_dict["subgroup_inf"] = batch.s
        return results_dict

    @implements(CdtModel)
    @torch.no_grad()
    def inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> Dict[str, float]:
        logits_all = aggregate_over_epoch(outputs=outputs, metric="logits")
        targets_all = aggregate_over_epoch(outputs=outputs, metric="targets")
        subgroup_inf_all = aggregate_over_epoch(outputs=outputs, metric="subgroup_inf")

        preds_all = prediction(logits_all)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(subgroup_inf_all).detach().cpu().numpy(), columns=["x0"]
            ),
            s=pd.DataFrame(subgroup_inf_all.detach().cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(targets_all.detach().cpu().numpy(), columns=["y"]),
        )

        return em.run_metrics(
            predictions=em.Prediction(hard=pd.Series(preds_all.detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )
