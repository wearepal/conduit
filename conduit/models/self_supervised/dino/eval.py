from __future__ import annotations
from collections import defaultdict
from dataclasses import replace

from kit import implements
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim

from conduit.architectures.vit import VisionTransformer
from conduit.data.structures import NamedSample, shallow_asdict
from conduit.models.erm import FineTuner
from conduit.models.utils import PartialModule

__all__ = [
    "DINOLinearClassifier",
    "DatasetEncoder",
]


class DINOLinearClassifier(FineTuner):
    encoder: nn.Module

    def __init__(
        self,
        encoder: nn.Module,
        *,
        embed_dim: int,
        target_dim: int,
        weight_decay: float,
        lr: float,
        epochs: int,
        num_eval_blocks: int = 1,
    ) -> None:
        self.epochs = epochs
        self.num_eval_blocks = num_eval_blocks
        if isinstance(encoder, VisionTransformer):
            embed_dim *= num_eval_blocks
            encoder = PartialModule(VisionTransformer.encode, num_eval_blocks=num_eval_blocks)
        classifier = nn.Linear(embed_dim, target_dim)
        super().__init__(
            encoder=encoder,
            classifier=classifier,
            lr=lr,
            weight_decay=weight_decay,
        )

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.CosineAnnealingLR]]:
        opt = optim.SGD(self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=self.epochs, eta_min=0)
        return [opt], [sched]


class DatasetEncoder(pl.LightningModule):
    """PyTorch-Lightning wrapper-class for dataset-encoding."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._dataset: NamedSample | None = None

    @property
    def dataset(self) -> NamedSample:
        if self._dataset is None:
            raise AttributeError(
                "Attribute 'dataset' cannot be accessed because the module has not yet been run."
            )
        return self._dataset

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @implements(pl.LightningModule)
    def test_step(self, batch: NamedSample, batch_idx: int) -> NamedSample:
        x = self(batch.x).detach()
        return replace(batch, x=x)

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[NamedSample]) -> None:
        cls = type(outputs[0])
        agg_dict = defaultdict(list)
        for step_output in outputs:
            for key, value in shallow_asdict(step_output):
                agg_dict[key].append(value)
        agg_dict = {key: torch.cat(value, dim=1) for key, value in agg_dict.items()}
        self._dataset = cls(**agg_dict)
