from abc import abstractmethod
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

import attr
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from ranzen.decorators import implements
from ranzen.misc import gcopy
from ranzen.torch.data import TrainingMode
import torch
from torch import nn
from torch.functional import Tensor
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
from conduit.models.self_supervised.multicrop import (
    MultiCropTransform,
    MultiCropWrapper,
)
from conduit.types import Stage

__all__ = [
    "BatchTransform",
    "InstanceDiscriminator",
    "MomentumTeacherModel",
    "SelfSupervisedModel",
]


@attr.define(kw_only=True, eq=False)
class SelfSupervisedModel(CdtModel):
    embed_dim: int = attr.field(init=False)
    eval_batch_size: Optional[int] = None
    eval_epochs: int = 100
    eval_lr: float = 3.0e-4

    _finetuner: Optional[pl.Trainer] = attr.field(default=None, init=False)
    _ft_clf: Optional[ERMClassifier] = attr.field(default=None, init=False)

    @abstractmethod
    def features(self, x: Tensor, **kwargs: Any) -> nn.Module:
        ...

    @property
    def ft_clf(self) -> ERMClassifier:
        if self._ft_clf is None:
            self._ft_clf = self._init_ft_clf()
            self._ft_clf.build(datamodule=self.datamodule, trainer=self.ft_trainer, copy=False)
        return self._ft_clf

    @ft_clf.setter
    def ft_clf(self, clf: ERMClassifier) -> None:
        self._ft_clf = clf

    @property
    def ft_trainer(self) -> pl.Trainer:
        if self._finetuner is None:
            self._finetuner = gcopy(self.trainer, deep=True, num_sanity_val_batches=0)
            self._finetuner.fit_loop.max_epochs = self.eval_epochs
            self._finetuner.fit_loop.max_steps = None  # type: ignore
            self._finetuner.logger = None  # type: ignore
            bar = PostHocProgressBar()
            bar._trainer = self._finetuner
            self._finetuner.callbacks = [bar]
        return self._finetuner

    @ft_trainer.setter
    def ft_trainer(self, trainer: pl.Trainer) -> None:
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

        self.ft_trainer.fit(
            self.ft_clf,
            train_dataloaders=dm_cp.train_dataloader(),
        )

    def on_inference_start(self) -> None:
        self._finetune()

    def validation_step(self, batch: NamedSample, batch_idx: int) -> STEP_OUTPUT:
        """Validation is handled entirely within on_validation_start/on_validation_end."""

    def test_step(self, batch: NamedSample, batch_idx: int) -> STEP_OUTPUT:
        """Testing is handled entirely within on_test_start/on_test_end."""

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results_dict = self.inference_epoch_end(outputs=outputs, stage=Stage.validate)
        self.log_dict(results_dict)  # type: ignore

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results_dict = self.inference_epoch_end(outputs=outputs, stage=Stage.test)
        self.log_dict(results_dict)  # type: ignore

    @implements(CdtModel)
    def inference_epoch_end(self, outputs: EPOCH_OUTPUT, stage: Stage) -> Dict[str, float]:
        if stage is Stage.validate:
            results_dict = self.ft_trainer.validate(self.ft_clf, self.datamodule)[0]
        else:
            results_dict = self.ft_trainer.test(self.ft_clf, self.datamodule)[0]
        self._ft_clf = None
        self.student.to(self.device)
        return results_dict

    @implements(CdtModel)
    def inference_step(self, batch: BinarySample, stage: Stage) -> STEP_OUTPUT:
        ...

    @implements(pl.LightningModule)
    def on_validation_start(self) -> None:
        self.on_inference_start()

    @implements(pl.LightningModule)
    def on_test_start(self) -> None:
        self.on_inference_start()


class BatchTransform(Protocol):
    def __call__(self, x: Tensor) -> Any:
        ...


@attr.define(kw_only=True, eq=False)
class InstanceDiscriminator(SelfSupervisedModel):
    instance_transforms: Optional[MultiCropTransform] = None
    batch_transforms: Optional[BatchTransform] = None
    _global_crop_size: Optional[Union[Tuple[int, int], int]] = None
    _local_crop_size: Union[Tuple[float, float], float] = 0.43
    global_crops_scale: Tuple[float, float] = (0.4, 1.0)
    local_crops_scale: Tuple[float, float] = (0.05, 0.4)
    local_crops_number: int = 0

    @property
    def global_crop_size(self) -> Union[int, Tuple[int, int]]:
        if not isinstance(self.datamodule, CdtVisionDataModule):
            raise AttributeError("'global_crop_size' is only applicable to vision datasets.")
        return (
            self.datamodule.dims[1:] if (self._global_crop_size is None) else self._global_crop_size
        )

    @property
    def local_crop_size(self) -> Union[int, Tuple[int, int]]:
        size_ = np.multiply(self.global_crop_size, self._local_crop_size).astype(np.int64)
        if isinstance(size_, np.integer):
            size = int(size_)
        else:
            size = cast(Tuple[int, int], tuple(size_))
        return size

    def _get_positives(self, batch: NamedSample) -> MultiCropOutput:
        if isinstance(batch.x, Tensor):
            if self.batch_transforms is None:
                return MultiCropOutput(global_crops=[batch.x, batch.x])
            view1, view2 = self.batch_transforms(torch.cat([batch.x, batch.x], dim=0)).chunk(
                2, dim=0
            )
            return MultiCropOutput(global_crops=[view1, view2])
        if isinstance(batch.x, MultiCropOutput):
            if self.batch_transforms is None:
                return batch.x
            global_crops = [self.batch_transforms(crop) for crop in batch.x.global_crops]
            local_crops = [self.batch_transforms(crop) for crop in batch.x.local_crops]
            return replace(batch.x, global_crops=global_crops, local_crops=local_crops)
        raise TypeError("'x' must be  a Tensor or a 'MultiCropTransform' instance.")

    @implements(CdtModel)
    def build(self, datamodule: CdtDataModule, *, trainer: pl.Trainer, copy: bool = True) -> None:
        super().build(datamodule=datamodule, trainer=trainer, copy=copy)
        if isinstance(datamodule, CdtVisionDataModule):
            datamodule.train_transforms = self.instance_transforms


@attr.define(kw_only=True, eq=False)
class MomentumTeacherModel(InstanceDiscriminator):
    student: MultiCropWrapper = attr.field(init=False)
    teacher: MultiCropWrapper = attr.field(init=False)

    @torch.no_grad()
    def init_encoders(self) -> Tuple[MultiCropWrapper, MultiCropWrapper]:
        student, teacher = self._init_encoders()
        # there is no backpropagation through the key-encoder, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        return student, teacher

    @torch.no_grad()
    @abstractmethod
    def _init_encoders(self) -> Tuple[MultiCropWrapper, MultiCropWrapper]:
        ...

    @property
    @abstractmethod
    def momentum_schedule(self) -> Union[float, np.ndarray, Tensor, Callable[[int], float]]:
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
