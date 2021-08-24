from __future__ import annotations
from typing import Sequence

from kit import implements
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

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
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.student_network = ...
        model.teacher_network = ...
        trainer = Trainer(callbacks=[MeanTeacherWeightUpdate])
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        momentum_schedule: np.ndarray | Tensor | float = 0.999,
    ) -> None:
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step.
        """
        super().__init__()

        self.student = student
        self.teacher = teacher
        self.momentum_schedule = momentum_schedule

    @implements(pl.Callback)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # update weights
        self.update(pl_module.global_step, student_net, teacher_net)  # type: ignore

    def update(self, train_itrs: int, student: nn.Module, teacher: nn.Module) -> None:
        # apply MA weight update
        if isinstance(self.momentum_schedule, np.ndarray):
            em = self.momentum_schedule[train_itrs]  # momentum parameter
        else:
            em = self.momentum_schedule
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
