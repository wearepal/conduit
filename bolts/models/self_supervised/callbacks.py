from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Callable, Sequence

from kit import implements
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBar, reset
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from tqdm import tqdm

if TYPE_CHECKING:
    from bolts.models.self_supervised.base import MomentumTeacherModel

__all__ = [
    "MeanTeacherWeightUpdate",
    "PostHocProgressBar",
]


class MeanTeacherWeightUpdate(pl.Callback):
    """
    Weight update rule from Mean Teacher.
    Your model should have:
        - ``self.student_network``
        - ``self.teacher_network``
    Updates the teacher_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    Example::
        # model must have 2 attributes
        model = Model()
        model.student_network = ...
        model.teacher_network = ...
        trainer = Trainer(callbacks=[MeanTeacherWeightUpdate])
    """

    def __init__(
        self, momentum_schedule: np.ndarray | Tensor | float | Callable[[int], float] = 0.999
    ) -> None:
        super().__init__()
        self.momentum_schedule = momentum_schedule

    @implements(pl.Callback)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: MomentumTeacherModel,
        outputs: STEP_OUTPUT,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.update_weights(train_step=pl_module.global_step, student=pl_module.student, teacher=pl_module.teacher)  # type: ignore

    def update_weights(self, *, train_step: int, student: nn.Module, teacher: nn.Module) -> None:
        # apply MA weight update
        if isinstance(self.momentum_schedule, np.ndarray):
            em = self.momentum_schedule[train_step]  # momentum parameter
        elif callable(self.momentum_schedule):
            em = self.momentum_schedule(train_step)
        else:
            em = self.momentum_schedule
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)


class PostHocProgressBar(ProgressBar):
    """Progress bar with descriptions tailored for post-hoc evaluation."""

    @implements(ProgressBar)
    def init_train_tqdm(self) -> tqdm:
        return tqdm(
            desc="Training (Post-Hoc)",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    @implements(ProgressBar)
    def init_validation_tqdm(self) -> tqdm:
        has_main_bar = self.main_progress_bar is not None
        return tqdm(
            desc="Validating (Post-Hoc)",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    @implements(ProgressBar)
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._train_batch_idx = 0
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float("inf") and total_val_batches != float("inf"):
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_batches = total_train_batches + total_val_batches
        reset(self.main_progress_bar, total=int(total_batches))
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch} (Post-Hoc Training)")
