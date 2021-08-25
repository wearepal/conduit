from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

from kit import implements
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

if TYPE_CHECKING:
    from bolts.models.self_supervised.base import SelfDistillation

__all__ = [
    "MeanTeacherWeightUpdate",
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

    def __init__(self, momentum_schedule: np.ndarray | Tensor | float = 0.999) -> None:
        super().__init__()
        self.momentum_schedule = momentum_schedule

    @implements(pl.Callback)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: SelfDistillation,
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
        else:
            em = self.momentum_schedule
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
