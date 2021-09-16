from typing import Any, Dict, Optional, Tuple, Union

from kit import implements, parsable
from kit.misc import gcopy, str_to_enum
from kit.torch.loss import CrossEntropyLoss, ReductionType
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F

from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import ImageTform
from conduit.data.structures import NamedSample
from conduit.models.base import CdtModel
from conduit.models.erm import FineTuner
from conduit.models.self_supervised.base import (
    BatchTransform,
    MomentumTeacherModel,
    SelfSupervisedModel,
)
from conduit.models.self_supervised.moco.transforms import (
    moco_ft_transform,
    moco_test_transform,
)
from conduit.models.self_supervised.moco.utils import (
    MemoryBank,
    ResNetArch,
    concat_all_gather,
)
from conduit.models.self_supervised.multicrop import (
    MultiCropTransform,
    MultiCropWrapper,
)
from conduit.models.utils import precision_at_k, prefix_keys
from conduit.types import MetricDict

__all__ = ["MoCoV2"]


class MoCoV2(MomentumTeacherModel):
    _ft_clf: FineTuner
    use_ddp: bool

    @parsable
    def __init__(
        self,
        *,
        backbone: Union[nn.Module, ResNetArch, str] = ResNetArch.resnet18,
        out_dim: int = 128,
        num_negatives: int = 65_536,
        momentum_teacher: float = 0.999,
        temp: float = 0.07,
        lr: float = 0.03,
        momentum_sgd: float = 0.9,
        weight_decay: float = 1.0e-4,
        use_mlp: bool = False,
        instance_transforms: Optional[MultiCropTransform] = None,
        batch_transforms: Optional[BatchTransform] = None,
        multicrop: bool = False,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
        eval_epochs: int = 100,
        eval_batch_size: Optional[int] = None,
    ) -> None:
        """
        PyTorch Lightning implementation of `MoCo <https://arxiv.org/abs/2003.04297>`_
        Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.

        :param backbone: Backbone of the encoder. Can be any nn.Module with a unary forward method
        (accepting a single input Tensor and returning a single output Tensor), in which case embed_dim
        will be inferred by passing a dummy input through the backone, or a 'ResNetArch' instance
        whose value is a resnet builder function which will be called with num_classes=out_dim.
        emb_dim: Feature dimension of the ResNet model: 128); only applicable if backbone is
        an 'ResNetArch' instance.

        :param out_dim: Output size of the encoder; only applicable when backbone is a 'ResNetArch' enum.

        :param num_negatives: queue size; number of negative keys.
        :param momentum_teacher: Momentum (what fraction of the previous iterates parameters to interpolate with)
        for the teacher update.

        :param temp: Softmax temperature.
        :param lr: Learning rate for the student model.
        :param sgd_momentum: Optimizer momentum.
        :param weight_decay: Optimizer weight decay.
        :param use_mlp: Whether to add an MLP head to the decoders (instead of a single linear layer).

        :param instance_transforms: Instance-wise image-transforms to use to generate the positive pairs
        for instance-discrimination.

        :param batch_transforms: Batch-wise image-transforms to use to generate the positive pairs for
        instance-discrimination.

        :param multicrop: Whether to use a multi-crop augmentation policy wherein the same image is
        randomly cropped to get a pair of high resolution (global) images and along with multiple
        lower resolution (generally covering less than 50% of the image) images of number 'local_crops_number'.

        :param global_crops_scale: Scale range of the cropped image before resizing, relative to the origin image.
        Used for large global view cropping. Only applies when 'multicrop=True'.

        :param local_crops_number: Number of small local views to generate.
        :param global_crops_scale: Scale range of the cropped image before resizing, relative to the origin image.
        Used for small, local cropping. Only applies when 'multicrop=True'.

        :param eval_epochs: Number of epochs to train the post-hoc classifier for during validation/testing.
        :param eval_batch_size: Batch size to use when training the post-hoc classifier during validation/testing.
        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            eval_epochs=eval_epochs,
            eval_batch_size=eval_batch_size,
            instance_transforms=instance_transforms,
            batch_transforms=batch_transforms,
        )
        if isinstance(backbone, str):
            backbone = str_to_enum(str_=backbone, enum=ResNetArch)
        self.backbone = backbone
        self.out_dim = out_dim
        self.temp = temp
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum_teacher = momentum_teacher
        self.momentum_sgd = momentum_sgd
        self.num_negatives = num_negatives
        self.use_mlp = use_mlp
        self.multicrop = multicrop
        self.local_crops_number = local_crops_number
        self.local_crops_scale = local_crops_scale
        self.global_crops_scale = global_crops_scale

        # create the queue
        self.mb = MemoryBank(dim=out_dim, capacity=num_negatives)
        self._loss_fn = CrossEntropyLoss(reduction=ReductionType.mean)

    @implements(CdtModel)
    def _build(self) -> None:
        self.use_ddp = "ddp" in str(self.trainer.distributed_backend)
        if isinstance(self.datamodule, CdtVisionDataModule):
            if (self.instance_transforms is None) and (self.batch_transforms is None):
                if self.multicrop:
                    self.instance_transforms = MultiCropTransform.with_dino_transform(
                        global_crop_size=224,
                        local_crop_size=96,
                        global_crops_scale=self.global_crops_scale,
                        local_crops_scale=self.local_crops_scale,
                        local_crops_number=self.local_crops_number,
                        norm_values=self.datamodule.norm_values,
                    )
                else:
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
    @implements(MomentumTeacherModel)
    def _init_encoders(self) -> Tuple[MultiCropWrapper, MultiCropWrapper]:
        # create the encoders
        if isinstance(self.backbone, ResNetArch):
            student_backbone = self.backbone.value(num_classes=self.out_dim)
            self.embed_dim = student_backbone.fc.weight.shape[1]
            head = student_backbone.fc
            student_backbone.fc = nn.Identity()
        else:
            student_backbone = self.backbone
            # Resort to computing embed_dim via the brute-force approach of passing in a dummy input.
            embed_dim_t = student_backbone(torch.zeros(1, *self.datamodule.size)).squeeze(0)
            # If the backbone does not produce a 1-dimensional embedding, add a flattening layer.
            if embed_dim_t.ndim > 1:
                student_backbone = nn.Sequential(student_backbone, nn.Flatten())
            self.embed_dim = embed_dim_t.numel()
            head = nn.Linear(self.embed_dim, self.out_dim)
        if self.use_mlp:
            head = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(), head)  # type: ignore

        student = MultiCropWrapper(backbone=student_backbone, head=head)
        teacher = gcopy(student, deep=True)

        return student, teacher

    @implements(SelfSupervisedModel)
    def features(self, x: Tensor, **kwargs: Any) -> nn.Module:
        return self.student(x, **kwargs)

    @property
    @implements(MomentumTeacherModel)
    def momentum_schedule(self) -> float:
        return self.momentum_teacher

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(
            self.student.parameters(),
            self.lr,
            momentum=self.momentum_sgd,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor) -> None:
        # gather keys before updating queue
        if self.use_ddp:
            keys = concat_all_gather(keys)
        self.mb.push(keys)

    @staticmethod
    @torch.no_grad()
    def _batch_shuffle_ddp(x: Tensor) -> Tuple[Tensor, Tensor]:  # pragma: no-cover
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

    @staticmethod
    @torch.no_grad()
    def _batch_unshuffle_ddp(x: Tensor, idx_unshuffle: Tensor) -> Tensor:  # pragma: no-cover
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
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, log_dict = self._inst_disc_loss(l_pos=l_pos, l_neg=l_neg)
        log_dict = prefix_keys(dict_=log_dict, prefix="inst_disc", sep="/")
        return loss, log_dict

    def _inst_disc_loss(self, l_pos: Tensor, l_neg: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
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
        global_crop_s, global_crop_t = views.global_crops
        # compute query features
        student_logits = self.student([global_crop_s] + views.local_crops)  # queries: NxC
        student_logits = F.normalize(student_logits, dim=1)

        # compute teacher's logits
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            idx_unshuffle = None
            if self.use_ddp:
                global_crop_t, idx_unshuffle = self._batch_shuffle_ddp(global_crop_t)

            teacher_logits = self.teacher(global_crop_t)  # keys: NxC
            teacher_logits = F.normalize(teacher_logits, dim=1)

            # undo shuffle
            if self.use_ddp:
                assert idx_unshuffle is not None
                teacher_logits = self._batch_unshuffle_ddp(teacher_logits, idx_unshuffle)

        # compute student's logits
        # positive logits: NxLx1
        student_logits_crop_view = student_logits.view(
            -1, global_crop_s.size(0), student_logits.size(-1)
        )
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
        assert isinstance(self.datamodule, CdtVisionDataModule)
        return moco_ft_transform(crop_size=224, norm_values=self.datamodule.norm_values)

    @implements(MomentumTeacherModel)
    @torch.no_grad()
    def _init_ft_clf(self) -> FineTuner:
        return FineTuner(
            encoder=self.student,
            classifier=nn.Linear(in_features=self.out_dim, out_features=self.datamodule.card_y),
        )
