from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from ranzen import implements
from ranzen.decorators import parsable
from ranzen.torch import CrossEntropyLoss, TrainingMode
from ranzen.torch.loss import ReductionType
import torch
from torch import Tensor, nn

from conduit.data import BinarySample
from conduit.metrics import accuracy, precision_at_k
from conduit.models.base import CdtModel
from conduit.models.utils import aggregate_over_epoch, make_no_grad, prefix_keys
from conduit.types import Loss, MetricDict, Stage

__all__ = ["ERMClassifier", "FineTuner"]


class ERMClassifier(CdtModel):
    @parsable
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
        loss_fn: Optional[Loss] = None,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
        )
        self.model = model
        self.loss_fn = (
            CrossEntropyLoss(reduction=ReductionType.mean) if loss_fn is None else loss_fn
        )

    def _get_loss(self, logits: Tensor, *, batch: BinarySample) -> Tensor:
        return self.loss_fn(input=logits, target=batch.y)

    @implements(pl.LightningModule)
    def training_step(self, batch: BinarySample, batch_idx: int) -> Tensor:
        assert isinstance(batch.x, Tensor)
        logits = self.forward(batch.x)
        loss = self._get_loss(logits=logits, batch=batch)
        results_dict = {
            "loss": loss.item(),
            "acc": accuracy(y_pred=logits, y_true=batch.y),
        }
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.fit), sep="/")
        self.log_dict(results_dict)

        return loss

    @implements(CdtModel)
    @torch.no_grad()
    def inference_step(self, batch: BinarySample, stage: Stage) -> Dict[str, Tensor]:
        assert isinstance(batch.x, Tensor)
        logits = self.forward(batch.x)
        return {"logits": logits, "targets": batch.y}

    @implements(CdtModel)
    @torch.no_grad()
    def inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> MetricDict:
        logits_all = aggregate_over_epoch(outputs=outputs, metric="logits")
        targets_all = aggregate_over_epoch(outputs=outputs, metric="targets")
        loss = self.loss_fn(input=logits_all, target=targets_all)

        results_dict: MetricDict = {"loss": loss}
        if logits_all.size(1) > 5:
            acc1, acc5 = precision_at_k(y_pred=logits_all, y_true=targets_all, top_k=(1, 5))
            results_dict["acc5"] = acc5
        else:
            acc1 = accuracy(y_pred=logits_all, y_true=targets_all)
        results_dict["acc1"] = acc1

        return results_dict

    @torch.no_grad()
    def _maybe_reset_parameters(self, module: nn.Module) -> None:
        if (module != self) and hasattr(module, 'reset_parameters'):
            module.reset_parameters()  # type: ignore

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.apply(self._maybe_reset_parameters)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class FineTuner(ERMClassifier):
    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
        loss_fn: Loss = CrossEntropyLoss(reduction="mean"),
    ) -> None:
        encoder = make_no_grad(encoder).eval()
        model = nn.Sequential(encoder, classifier)
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
        self.classifier = classifier
