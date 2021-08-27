from __future__ import annotations
from typing import Any, Callable, Optional, Union

from kit import implements, parsable
from kit.misc import gcopy
from kit.torch.loss import CrossEntropyLoss, ReductionType
import pytorch_lightning as pl
import torch
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as F

from bolts.data.datamodules.vision.base import PBVisionDataModule
from bolts.data.datasets.utils import ImageTform
from bolts.data.structures import NamedSample
from bolts.models.base import PBModel
from bolts.models.erm import FineTuner
from bolts.models.self_supervised.base import SelfDistiller, SelfSupervisedModel
from bolts.models.self_supervised.moco.transforms import (
    moco_ft_transform,
    moco_test_transform,
)
from bolts.models.self_supervised.multicrop import MultiCropTransform, MultiCropWrapper
from bolts.models.utils import precision_at_k, prefix_keys
from bolts.types import MetricDict

from .utils import MemoryBank, ResNetArch, concat_all_gather

__all__ = ["MoCoV2"]


class MoCoV2(SelfDistiller):
    ft_clf: FineTuner
    use_ddp: bool

    @parsable
    def __init__(
        self,
        *,
        backbone: Union[nn.Module, ResNetArch] = ResNetArch.resnet18,
        out_dim: int = 128,
        num_negatives: int = 65_536,
        momentum_teacher: float = 0.999,
        temp: float = 0.07,
        lr: float = 0.03,
        momentum_sgd: float = 0.9,
        weight_decay: float = 1.0e-4,
        use_mlp: bool = False,
        eval_epochs: int = 100,
        eval_batch_size: Optional[int] = None,
        instance_transforms: Optional[MultiCropTransform] = None,
        batch_transforms: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        """
        PyTorch Lightning implementation of `MoCo <https://arxiv.org/abs/2003.04297>`_
        Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.
        Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`

        Args:
            arch: ResNet architecture to use for the encoders.
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            temp: softmax temperature (default: 0.07)
            lr: the learning rate
            sgd_momentum: optimizer momentum
            weight_decay: optimizer weight decay
            use_mlp: add an mlp to the encoders
        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            eval_epochs=eval_epochs,
            eval_batch_size=eval_batch_size,
            instance_transforms=instance_transforms,
            batch_transforms=batch_transforms,
        )
        self.backbone = backbone
        self.out_dim = out_dim
        self.temp = temp
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum_teacher = momentum_teacher
        self.momentum_sgd = momentum_sgd

        self.num_negatives = num_negatives
        # create the queue
        self.mb = MemoryBank(dim=out_dim, capacity=num_negatives)
        self.use_mlp = use_mlp
        self._loss_fn = CrossEntropyLoss(reduction=ReductionType.mean)

    @implements(PBModel)
    def _build(self) -> None:
        self.use_ddp = "ddp" in str(self.trainer.distributed_backend)
        if isinstance(self.datamodule, PBVisionDataModule):
            # self._datamodule.train_transforms = mocov2_transform()
            if (self.instance_transforms is None) and (self.batch_transforms is None):
                self.instance_transforms = MultiCropTransform.with_mocov2_transform(
                    crop_size=224,
                    norm_values=self.datamodule.norm_values,
                )
            self.datamodule.test_transforms = moco_test_transform(
                crop_size=224,
                amount_to_crop=32,
                norm_values=self.datamodule.norm_values,
            )

    @torch.no_grad()
    @implements(SelfDistiller)
    def _init_encoders(self) -> tuple[MultiCropWrapper, MultiCropWrapper]:
        # create the encoders
        if isinstance(self.backbone, ResNetArch):
            student_backbone = self.backbone.value(num_classes=self.out_dim)
            self.embed_dim = student_backbone.fc.weight.shape[1]
            head = student_backbone.fc
            student_backbone.fc = nn.Identity()
        else:
            student_backbone = self.backbone
            # Brute-force computation using a dummy input
            embed_dim_t = student_backbone(torch.zeros(1, *self.datamodule.size)).squeeze(0)
            # If the backbone does not produce a 1-dimensional embedding, add a flattening layer
            if embed_dim_t.ndim > 1:
                student_backbone = nn.Sequential(student_backbone, nn.Flatten())
            self.embed_dim = embed_dim_t.numel()
            head = (
                nn.Identity()
                if self.embed_dim == self.out_dim
                else nn.Linear(self.embed_dim, self.out_dim)
            )
        if self.use_mlp:
            head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), head)  # type: ignore
        student = MultiCropWrapper(backbone=student_backbone, head=head)

        teacher = gcopy(student, deep=True)

        return student, teacher

    @implements(SelfSupervisedModel)
    def features(self, x: Tensor, **kwargs: Any) -> nn.Module:
        return self.student(x, **kwargs)

    @property
    @implements(SelfDistiller)
    def momentum_schedule(self) -> float:
        return self.momentum_teacher

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.SGD(
            self.student.parameters(),
            self.lr,
            momentum=self.momentum_sgd,
            weight_decay=self.weight_decay,
        )
        return optimizer

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor) -> None:
        # gather keys before updating queue
        if self.use_ddp:
            keys = concat_all_gather(keys)
        self.mb.push(keys)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: Tensor) -> tuple[Tensor, Tensor]:  # pragma: no-cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only supports DistributedDataParallel (DDP).***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)  # type: ignore

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()  # type: ignore
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: Tensor, idx_unshuffle: Tensor) -> Tensor:  # pragma: no-cover
        """
        Undo batch shuffle.
        *** Only supports DistributedDataParallel (DDP).***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()  # type: ignore
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def _get_loss(
        self,
        *,
        l_pos: Tensor,
        l_neg: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss, log_dict = self._inst_disc_loss(l_pos=l_pos, l_neg=l_neg)
        log_dict = prefix_keys(dict_=log_dict, prefix="inst_disc", sep="/")
        return loss, log_dict

    def _inst_disc_loss(self, l_pos: Tensor, l_neg: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.temp
        targets = logits.new_zeros(size=(logits.size(0),))
        loss = self._loss_fn(input=logits, target=targets)
        acc1, acc5 = precision_at_k(logits, targets, top_k=(1, 5))

        return loss, {'loss': loss.detach(), 'acc1': acc1, 'acc5': acc5}

    @implements(pl.LightningModule)
    def training_step(self, batch: NamedSample, batch_idx: int) -> MetricDict:
        views = self._get_positive_views(batch=batch)
        img_q, img_k = views.global_crops
        # compute query features
        student_logits = self.student([img_q] + views.local_crops)  # queries: NxC
        student_logits = F.normalize(student_logits, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            idx_unshuffle = None
            if self.use_ddp:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            teacher_logits = self.teacher(img_k)  # keys: NxC
            teacher_logits = F.normalize(teacher_logits, dim=1)

            # undo shuffle
            if self.use_ddp:
                assert idx_unshuffle is not None
                teacher_logits = self._batch_unshuffle_ddp(teacher_logits, idx_unshuffle)

        # compute logits
        # positive logits: NxLx1
        student_logits_crop_view = student_logits.view(-1, img_q.size(0), student_logits.size(-1))
        l_pos = (student_logits_crop_view * teacher_logits.unsqueeze(0)).sum(-1).view(-1, 1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,kc->nk', [student_logits, self.mb.memory.clone()])

        loss, log_dict = self._get_loss(l_pos=l_pos, l_neg=l_neg)

        # dequeue and enqueue
        self._dequeue_and_enqueue(teacher_logits)

        return {'loss': loss, 'log': log_dict, 'progress_bar': log_dict}

    @property
    @implements(SelfSupervisedModel)
    def _ft_transform(self) -> ImageTform:
        assert isinstance(self.datamodule, PBVisionDataModule)
        return moco_ft_transform(crop_size=224, norm_values=self.datamodule.norm_values)

    @implements(SelfDistiller)
    @torch.no_grad()
    def _init_ft_clf(self) -> FineTuner:
        return FineTuner(
            encoder=self.student,
            classifier=nn.Linear(in_features=self.out_dim, out_features=self.datamodule.card_y),
        )
