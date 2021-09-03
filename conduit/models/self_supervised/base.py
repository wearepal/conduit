from __future__ import annotations
from abc import abstractmethod
from dataclasses import replace
from typing import Any, Callable, Optional

from kit.decorators import implements
from kit.misc import gcopy
from kit.torch.data import TrainingMode
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch.functional import Tensor
import torch.nn as nn
from typing_extensions import Protocol

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import ImageTform
from conduit.data.structures import BinarySample, MultiCropOutput, NamedSample
from conduit.models.base import CdtModel
from conduit.models.erm import ERMClassifier
from conduit.models.self_supervised.callbacks import (
    MeanTeacherWeightUpdate,
    PostHocProgressBar,
)
from conduit.models.self_supervised.multicrop import MultiCropTransform, MultiCropWrapper
from conduit.types import MetricDict, Stage

__all__ = [
    "BatchTransform",
    "InstanceDiscriminator",
    "MomentumTeacherModel",
    "SelfSupervisedModel",
]


class SelfSupervisedModel(CdtModel):
    embed_dim: int

    def __init__(
        self,
        *,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        eval_batch_size: int | None = None,
        eval_epochs: int = 100,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.eval_batch_size = eval_batch_size
        self.eval_epochs = eval_epochs
        self._finetuner: pl.Trainer | None = None
        self._ft_clf: ERMClassifier | None = None

    @abstractmethod
    def features(self, x: Tensor, **kwargs: Any) -> nn.Module:
        ...

    @property
    def ft_clf(self) -> ERMClassifier:
        if self._ft_clf is None:
            self._ft_clf = self._init_ft_clf()
            self._ft_clf.build(datamodule=self.datamodule, trainer=self.finetuner, copy=False)
        return self._ft_clf

    @ft_clf.setter
    def ft_clf(self, clf: ERMClassifier) -> None:
        self._ft_clf = clf

    @property
    def finetuner(self) -> pl.Trainer:
        if self._finetuner is None:
            self._finetuner = gcopy(self.trainer, deep=True, num_sanity_val_batches=0)
            self._finetuner.fit_loop.max_epochs = self.eval_epochs
            self._finetuner.fit_loop.max_steps = None  # type: ignore
            bar = PostHocProgressBar()
            bar._trainer = self._finetuner
            self._finetuner.callbacks = [bar]
        return self._finetuner

    @finetuner.setter
    def finetuner(self, trainer: pl.Trainer) -> None:
        self._finetuner = trainer

    @abstractmethod
    def _init_ft_clf(self) -> ERMClassifier:
        ...

    @property
    @abstractmethod
    def _ft_transform(self) -> ImageTform:
        ...

    def _finetune(self) -> None:
        dm_cp = gcopy(
            self.datamodule,
            deep=False,
            stratified_sampling=False,
            training_mode=TrainingMode.epoch,
        )
        if isinstance(dm_cp, CdtVisionDataModule):
            dm_cp.train_transforms = self._ft_transform
        if self.eval_batch_size is not None:
            dm_cp.train_batch_size = self.eval_batch_size

        self.finetuner.fit(
            self.ft_clf,
            train_dataloaders=dm_cp.train_dataloader(),
        )

    @implements(CdtModel)
    def inference_step(self, batch: BinarySample, stage: Stage) -> STEP_OUTPUT:
        return self.ft_clf.inference_step(batch=batch, stage=stage)

    @implements(CdtModel)
    def inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> MetricDict:
        results_dict = self.ft_clf.inference_epoch_end(outputs=outputs, stage=stage)
        # Free up memory
        self._ft_clf = None
        return results_dict

    def on_inference_start(self) -> None:
        self._finetune()

    @implements(pl.LightningModule)
    def on_validation_start(self) -> None:
        self.on_inference_start()

    @implements(pl.LightningModule)
    def on_test_start(self) -> None:
        self.on_inference_start()


class BatchTransform(Protocol):
    def __call__(self, Tensor) -> Any:
        ...


class InstanceDiscriminator(SelfSupervisedModel):
    def __init__(
        self,
        *,
        lr: float,
        weight_decay: float,
        eval_batch_size: int | None,
        eval_epochs: int,
        instance_transforms: MultiCropTransform | None = None,
        batch_transforms: Optional[BatchTransform] = None,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            eval_batch_size=eval_batch_size,
            eval_epochs=eval_epochs,
        )
        self.instance_transforms = instance_transforms
        self.batch_transforms = batch_transforms

    def _get_positive_views(self, batch: NamedSample) -> MultiCropOutput:
        if isinstance(batch.x, Tensor):
            if self.batch_transforms is None:
                return MultiCropOutput(global_crops=[batch.x, batch.x])
            view1, view2 = self.batch_transforms(torch.cat([batch.x, batch.x], dim=0)).chunk(
                2, dim=0
            )
            return MultiCropOutput(global_crops=[view1, view2])
        elif isinstance(batch.x, MultiCropOutput):
            if self.batch_transforms is None:
                return batch.x
            global_crops = [self.batch_transforms(crop) for crop in batch.x.global_crops]
            local_crops = [self.batch_transforms(crop) for crop in batch.x.local_crops]
            return replace(batch.x, global_crops=global_crops, local_crops=local_crops)
        else:
            raise TypeError("'x' must be  a Tensor or a 'MultiCropTransform' instance.")

    @implements(CdtModel)
    def build(self, datamodule: CdtDataModule, *, trainer: pl.Trainer, copy: bool = True) -> None:
        super().build(datamodule=datamodule, trainer=trainer, copy=copy)
        if isinstance(datamodule, CdtVisionDataModule):
            datamodule.train_transforms = self.instance_transforms


class MomentumTeacherModel(InstanceDiscriminator):
    student: MultiCropWrapper
    teacher: MultiCropWrapper

    @torch.no_grad()
    def init_encoders(self) -> tuple[MultiCropWrapper, MultiCropWrapper]:
        student, teacher = self._init_encoders()
        # there is no backpropagation through the key-encoder, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        return student, teacher

    @torch.no_grad()
    @abstractmethod
    def _init_encoders(self) -> tuple[MultiCropWrapper, MultiCropWrapper]:
        ...

    @property
    @abstractmethod
    def momentum_schedule(self) -> float | np.ndarray | Tensor | Callable[[int], float]:
        ...

    @implements(InstanceDiscriminator)
    def build(self, datamodule: CdtDataModule, *, trainer: pl.Trainer, copy: bool = True) -> None:
        super().build(datamodule=datamodule, trainer=trainer, copy=copy)
        self.student, self.teacher = self.init_encoders()
        mt_cb = MeanTeacherWeightUpdate(momentum_schedule=self.momentum_schedule)
        if self.trainer.callbacks is None:
            self.trainer.callbacks = [mt_cb]
        else:
            self.trainer.callbacks.append(mt_cb)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.student(x)
