from typing import TYPE_CHECKING, Callable, Sequence, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from ranzen import implements
from torch import Tensor, nn

if TYPE_CHECKING:
    from conduit.models.self_supervised.base import MomentumTeacherModel

__all__ = ["MeanTeacherWeightUpdate"]


class MeanTeacherWeightUpdate(pl.Callback):
    """
    Weight update rule from Mean Teacher.

    Updates the teacher_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    """

    def __init__(
        self, momentum_schedule: Union[np.ndarray, Tensor, float, Callable[[int], float]] = 0.999
    ) -> None:
        super().__init__()
        self.momentum_schedule = momentum_schedule

    @implements(pl.Callback)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "MomentumTeacherModel",
        outputs: STEP_OUTPUT,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.update_weights(
            train_step=pl_module.global_step, student=pl_module.student, teacher=pl_module.teacher  # type: ignore
        )

    def update_weights(self, *, train_step: int, student: nn.Module, teacher: nn.Module) -> None:
        # apply MA weight update
        if isinstance(self.momentum_schedule, np.ndarray):
            em = self.momentum_schedule[train_step]  # momentum parameter
        elif callable(self.momentum_schedule):
            em = self.momentum_schedule(train_step)
        else:
            em = self.momentum_schedule
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            param_t.data = param_t.data * em + param_s.data * (1.0 - em)
