from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

from kit import implements
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from .utils import cosine_scheduler

if TYPE_CHECKING:
    from .lightning import DINO

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

    def __init__(self, max_steps: int, initial_tau: float = 0.996) -> None:
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step.
        """
        super().__init__()

        self.momentum_schedule = cosine_scheduler(
            base_value=initial_tau,
            final_value=1,
            total_iters=max_steps,
        )

    @implements(pl.Callback)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: DINO,
        outputs: STEP_OUTPUT,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # get networks
        student_net = pl_module.student
        teacher_net = pl_module.teacher

        # update weights
        self.update_weights(pl_module.global_step, student_net, teacher_net)  # type: ignore

    def update_weights(self, train_itrs: int, student: nn.Module, teacher: nn.Module) -> None:
        # apply MA weight update
        em = self.momentum_schedule[train_itrs]  # momentum parameter
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
