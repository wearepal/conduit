from typing import Dict

import attr
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from ranzen import implements
from ranzen.torch import CrossEntropyLoss
import torch
from torch import Tensor, nn

from conduit.data import BinarySample
from conduit.models.base import CdtModel
from conduit.models.utils import (
    accuracy,
    aggregate_over_epoch,
    make_no_grad,
    precision_at_k,
    prefix_keys,
)
from conduit.types import Loss, MetricDict, Stage

__all__ = ["ERMClassifier", "FineTuner"]


@attr.define(kw_only=True, eq=False)
class ERMClassifier(CdtModel):
    model: nn.Module
    loss_fn: Loss = attr.field(factory=CrossEntropyLoss)

    def _get_loss(self, logits: Tensor, *, batch: BinarySample) -> Tensor:
        return self.loss_fn(input=logits, target=batch.y)

    @implements(pl.LightningModule)
    def training_step(self, batch: BinarySample, batch_idx: int) -> Tensor:
        assert isinstance(batch.x, Tensor)
        logits = self.forward(batch.x)
        loss = self._get_loss(logits=logits, batch=batch)
        results_dict = {
            "loss": loss.item(),
            "acc": accuracy(logits=logits, targets=batch.y),
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
            acc1, acc5 = precision_at_k(logits=logits_all, targets=targets_all, top_k=(1, 5))
            results_dict["acc5"] = acc5
        else:
            acc1 = accuracy(logits=logits_all, targets=targets_all)
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


@attr.define(kw_only=True, eq=False)
class FineTuner(ERMClassifier):
    classifier: nn.Module
    encoder: nn.Module
    model: None = None

    def __attrs_post_init__(self) -> None:
        self.encoder = make_no_grad(self.encoder).eval()
        super().__init__(model=nn.Sequential(self.encoder, self.classifier))
        breakpoint()
