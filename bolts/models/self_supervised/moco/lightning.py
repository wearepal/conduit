from __future__ import annotations
from typing import Protocol, Sequence, cast

from kit import implements, parsable
from kit.misc import gcopy
from kit.torch.loss import CrossEntropyLoss, ReductionType
from pl_bolts.metrics import mean
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from bolts.callbacks.posthoc_eval import PostHocEval
from bolts.data.datamodules.base import PBDataModule
from bolts.data.datamodules.vision.base import PBVisionDataModule
from bolts.data.structures import NamedSample
from bolts.models.self_supervised.moco.transforms import TwoCropsTransform
from bolts.models.utils import precision_at_k
from bolts.types import MetricDict

from .callbacks import MeanTeacherWeightUpdate
from .utils import MemoryBank, ResNetArch, concat_all_gather

__all__ = ["MoCoV2"]


class EncoderFn(Protocol):
    def __call__(self, **kwargs) -> resnet.ResNet:
        ...


class MoCoV2(pl.LightningModule):
    @parsable
    def __init__(
        self,
        *,
        arch: ResNetArch = ResNetArch.resnet18,
        emb_dim: int = 128,
        num_negatives: int = 65_536,
        encoder_momentum: float = 0.999,
        temp: float = 0.07,
        lr: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1.0e-4,
        use_mlp: bool = False,
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
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            use_mlp: add an mlp to the encoders
        """
        super().__init__()
        self._arch_fn = cast(EncoderFn, arch.value)
        self.emb_dim = emb_dim
        self.temp = temp
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        # create the encoders
        # num_classes is the output fc dimension
        self.student, self.teacher = self._init_encoders()

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.student.fc.weight.shape[1]
            self.student.fc = nn.Sequential(  # type: ignore
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.student.fc
            )
            self.teacher.fc = nn.Sequential(  # type: ignore
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.teacher.fc
            )
        self.momentum_update = MeanTeacherWeightUpdate(em=encoder_momentum)

        self.num_negatives = num_negatives
        # create the queue
        self.mb = MemoryBank(dim=emb_dim, capacity=num_negatives)
        self._loss_fn = CrossEntropyLoss(reduction=ReductionType.mean)

    def build(self, *, datamodule: PBDataModule, trainer: pl.Trainer) -> None:
        self._datamodule = datamodule
        if isinstance(self._datamodule, PBVisionDataModule):
            # self._datamodule.train_transforms = mocov2_transform()
            self._datamodule.train_transforms = TwoCropsTransform.with_mocov2_transform()
        self._trainer = gcopy(trainer)
        self.lin_eval_trainer = gcopy(trainer)
        # self.lin_eval_trainer.callbacks.append(PostHocEval)

    @torch.no_grad()
    def _init_encoders(self) -> tuple[resnet.ResNet, resnet.ResNet]:
        encoder_q = self._arch_fn(num_classes=self.emb_dim)
        encoder_k = self._arch_fn(num_classes=self.emb_dim)

        # key and query encoders start with the same weights
        encoder_k.load_state_dict(encoder_q.state_dict())  # type: ignore
        # there is no backpropagation through the key-encoder, so no need for gradients
        for p in encoder_k.parameters():
            p.requires_grad = False

        return encoder_q, encoder_k

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.SGD(
            self.parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        return optimizer

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor) -> None:
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)
        self.mb.push(keys)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: Tensor) -> tuple[Tensor, Tensor]:  # pragma: no-cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only supports DistributedDataParallel (DDP) model. ***
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
        *** Only support DistributedDataParallel (DDP) model. ***
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

    def forward(self, img_q: Tensor, img_k: Tensor) -> Tensor:
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.student(img_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            idx_unshuffle = None
            if self.use_ddp or self.use_ddp2:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.teacher(img_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            # undo shuffle
            if self.use_ddp or self.use_ddp2:
                assert idx_unshuffle is not None
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,kc->nk', [q, self.mb.memory])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temp

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits

    def training_step(self, batch: NamedSample, batch_idx: int) -> MetricDict:
        img_1, img_2 = batch.x
        logits = self.forward(img_q=img_1, img_k=img_2)
        targets = logits.new_zeros(size=(logits.size(0),))
        loss = self._loss_fn(input=logits, targets=targets)
        acc1, acc5 = precision_at_k(logits, targets, top_k=(1, 5))

        log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
        return {'loss': loss, 'log': log, 'progress_bar': log}

    @implements(pl.LightningModule)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        outputs: STEP_OUTPUT,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.momentum_update.update_weights(student=self.student, teacher=self.teacher)

    def validation_step(self, batch: NamedSample, batch_idx: int) -> MetricDict:
        breakpoint()
        output, target = self.forward(*batch.x)
        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        return {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> MetricDict:
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {'val_loss': val_loss, 'val_acc1': val_acc1, 'val_acc5': val_acc5}
        return {'val_loss': val_loss, 'log': log, 'progress_bar': log}
