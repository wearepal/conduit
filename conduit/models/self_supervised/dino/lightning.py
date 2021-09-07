from __future__ import annotations
from typing import Any, Callable, Optional, Tuple, Union, cast

from kit import gcopy, implements, parsable
from kit.misc import str_to_enum
from kit.torch.data import TrainingMode
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet

from conduit.architectures import vit
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import ImageTform
from conduit.data.structures import NamedSample
from conduit.models.base import CdtModel
from conduit.models.self_supervised.base import (
    BatchTransform,
    MomentumTeacherModel,
    SelfSupervisedModel,
)
from conduit.models.self_supervised.dino.callbacks import DINOScheduler
from conduit.models.self_supervised.dino.eval import DINOLinearClassifier
from conduit.models.self_supervised.dino.head import DINOHead
from conduit.models.self_supervised.dino.loss import DINOLoss
from conduit.models.self_supervised.dino.transforms import MultiCropTransform
from conduit.models.self_supervised.dino.utils import (
    cosine_scheduler,
    get_params_groups,
)
from conduit.models.self_supervised.moco.transforms import (
    moco_ft_transform,
    moco_test_transform,
)
from conduit.models.self_supervised.moco.utils import ResNetArch
from conduit.models.self_supervised.multicrop import MultiCropWrapper
from conduit.types import Stage

__all__ = ["DINO"]


class DINO(MomentumTeacherModel):
    _ft_clf: DINOLinearClassifier

    @parsable
    def __init__(
        self,
        *,
        backbone: Union[
            nn.Module, vit.VitArch, vit.VisionTransformer, ResNetArch, str
        ] = vit.VitArch.small,
        out_dim: int = 65_536,
        lr: float = 5.0e-4,
        warmup_iters: int = 10,
        weight_decay: float = 4.0e-2,
        min_lr: float = 1.0e-6,
        weight_decay_final: float = 0.4,
        freeze_last_layer: int = 1,
        patch_size: int = 16,
        drop_path_rate: float = 0.1,
        norm_last_layer: bool = True,
        use_bn_in_head: bool = False,
        momentum_teacher: float = 0.996,
        momentum_center: float = 0.9,
        teacher_temp: float = 0.04,
        warmup_teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        warmup_teacher_temp_iters: int = 30,
        num_eval_blocks: int = 1,
        lr_eval: float = 1.0e-4,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
        batch_transforms: Optional[BatchTransform] = None,
        eval_epochs: int = 100,
        eval_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            eval_epochs=eval_epochs,
            eval_batch_size=eval_batch_size,
            batch_transforms=batch_transforms,
        )
        if isinstance(backbone, str):
            backbone = str_to_enum(str_=backbone, enum=vit.VitArch)
        self.backbone = backbone
        self.num_eval_blocks = num_eval_blocks
        self.warmup_iters = warmup_iters
        self.min_weight_decay = weight_decay_final
        self.min_lr = min_lr
        self.freeze_last_layer = freeze_last_layer

        self.out_dim = out_dim
        self.norm_last_layer = norm_last_layer
        self.use_bn_in_head = use_bn_in_head
        self.momentum_teacher = momentum_teacher
        self.momentum_center = momentum_center
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_iters = warmup_teacher_temp_iters
        self.student_temp = student_temp
        self.eval_lr = lr_eval
        self.local_crops_number = local_crops_number
        self.local_crops_scale = local_crops_scale
        self.global_crops_scale = global_crops_scale

        # ViT-specific arguments
        self.patch_size = patch_size
        self.drop_path_rate = drop_path_rate

        self._loss_fn = DINOLoss(
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp=warmup_teacher_temp,
            warmup_teacher_temp_iters=warmup_teacher_temp_iters,
        )

    @implements(CdtModel)
    def _build(self) -> None:
        if isinstance(self.datamodule, CdtVisionDataModule):
            self.instance_transforms = MultiCropTransform.with_dino_transform(
                global_crop_size=224,
                local_crop_size=96,
                global_crops_scale=self.global_crops_scale,
                local_crops_scale=self.local_crops_scale,
                local_crops_number=self.local_crops_number,
                norm_values=self.datamodule.norm_values,
            )

            self.datamodule.test_transforms = moco_test_transform(
                crop_size=224,
                amount_to_crop=32,
                norm_values=self.datamodule.norm_values,
            )

        scheduler_cb = DINOScheduler(
            base_lr=self.lr * self.datamodule.train_batch_size / 256.0,  # linear scaling rule
            min_lr=self.min_lr,
            base_wd=self.weight_decay,
            min_wd=self.min_weight_decay,
            total_iters=self.num_training_steps,
        )
        self.trainer.callbacks.append(scheduler_cb)

    @implements(SelfSupervisedModel)
    def features(self, x: Tensor, **kwargs: Any) -> nn.Module:
        return self.student.backbone(x, **kwargs)

    @property
    @implements(MomentumTeacherModel)
    def momentum_schedule(self) -> np.ndarray:
        return cosine_scheduler(
            base_value=self.momentum_teacher,
            final_value=1,
            total_iters=self.num_training_steps,
        )

    @torch.no_grad()
    @implements(MomentumTeacherModel)
    def _init_encoders(self) -> tuple[MultiCropWrapper, MultiCropWrapper]:
        if isinstance(self.backbone, vit.VitArch):
            self.backbone = cast(
                vit.VisionTransformer,
                self.backbone.value(self.patch_size, drop_path_rate=self.drop_path_rate),
            )
        if isinstance(self.backbone, vit.VisionTransformer):
            student_backbone = self.backbone
            self.embed_dim = student_backbone.embed_dim
            # disable layers dedicated to ImageNet classification
            student_backbone.fc = nn.Identity()
            student_backbone.head = nn.Identity()
        elif isinstance(self.backbone, ResNetArch):
            student_backbone = cast(ResNet, self.backbone.value(self.patch_size))
            self.embed_dim = student_backbone.fc.weight.shape[1]
            student_backbone.fc = nn.Identity()  # type: ignore
        else:
            student_backbone = self.backbone
            embed_dim_t = student_backbone(torch.zeros(1, *self.datamodule.size)).squeeze(0)
            # If the backbone does not produce a 1-dimensional embedding, add a flattening layer
            if embed_dim_t.ndim > 1:
                student_backbone = nn.Sequential(student_backbone, nn.Flatten())
            self.embed_dim = embed_dim_t.numel()
        teacher_backbone = gcopy(student_backbone, deep=True)

        student_head = DINOHead(
            in_dim=self.embed_dim,
            out_dim=self.out_dim,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
        )
        teacher_head = gcopy(student_head, deep=True)
        teacher_head.last_layer.weight_g.requires_grad = False

        student = MultiCropWrapper(backbone=student_backbone, head=student_head)
        teacher = MultiCropWrapper(backbone=teacher_backbone, head=teacher_head)

        return student, teacher

    @torch.no_grad()
    def _encode_dataset(self, stage: Stage) -> NamedSample:
        # It's not strictly necessary to disable shuffling but pytorch-lightning complains if its
        # enabled during 'testing'
        dl_kwargs = (
            dict(shuffle=False, train_batch_size=self.datamodule.eval_batch_size)
            if stage == "train"
            else {}
        )
        train_transform = self._ft_transform
        dm_cp = gcopy(
            self.datamodule,
            deep=False,
            stratified_sampling=False,
            training_mode=TrainingMode.epoch,
            train_transforms=train_transform,
        )
        dataloader = cast(DataLoader, getattr(dm_cp, f"{stage}_dataloader")(**dl_kwargs))
        # Encode the dataset
        dataset_encoder = DatasetEncoder(model=self.student.backbone)
        self.finetuner.test(
            dataset_encoder,
            test_dataloaders=dataloader,
            verbose=False,
        )
        # Extract the encodings
        return dataset_encoder.dataset

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(
            get_params_groups(self.student), lr=self.lr, weight_decay=self.weight_decay
        )

    def _cancel_gradients_last_layer(self, train_itr: int) -> None:
        if train_itr >= self.freeze_last_layer:
            return
        for n, p in self.student.named_parameters():
            if "last_layer" in n:
                p.grad = None

    def _get_loss(self, batch: NamedSample, batch_idx: int) -> Tensor:
        views = self._get_positive_views(batch=batch)
        teacher_logits = self.teacher(
            views.global_crops
        )  # only the 2 global views pass through the teacher
        student_logits = self.student(views.all_crops)
        return self._loss_fn(
            student_logits=student_logits, teacher_logits=teacher_logits, step=batch_idx
        )

    @implements(pl.LightningModule)
    def training_step(self, batch: NamedSample, batch_idx: int) -> Tensor:
        return self._get_loss(batch=batch, batch_idx=batch_idx)

    @implements(pl.LightningModule)
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure: Callable | None,
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool,
    ) -> None:
        # Keep the output layer fixed until the epoch-threshold has been reached
        # Typically doing so during the first epoch helps training.
        self._cancel_gradients_last_layer(train_itr=batch_idx)
        # Update the student's parameters using the DINO loss
        super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
        )

    @property
    @implements(SelfSupervisedModel)
    def _ft_transform(self) -> ImageTform:
        assert isinstance(self.datamodule, CdtVisionDataModule)
        return moco_ft_transform(crop_size=224, norm_values=self.datamodule.norm_values)

    @implements(SelfSupervisedModel)
    @torch.no_grad()
    def _init_ft_clf(self) -> DINOLinearClassifier:
        return DINOLinearClassifier(
            encoder=self.student.backbone,
            embed_dim=self.embed_dim,
            target_dim=self.datamodule.card_y,
            epochs=self.eval_epochs,
            weight_decay=0,
            lr=self.eval_lr,
        )
