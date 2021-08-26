from __future__ import annotations

from kit.decorators import implements
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DINOLoss", "EMACenter"]


class DINOLoss(nn.Module):

    center: Tensor

    def __init__(
        self,
        out_dim: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_iters: int,
        total_iters: int,
        num_crops: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_crops = num_crops
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_iters),
                np.ones(total_iters - warmup_teacher_temp_iters) * teacher_temp,
            )
        )

    def forward(self, student_output: Tensor, teacher_output: Tensor, step: int) -> Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.num_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[step]
        teacher_out = ((teacher_output - self.center) / temp).softmax(dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = student_out[-1].new_zeros(())
        n_loss_terms = student_out[-1].new_zeros(())
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class EMACenter(nn.Module):
    def __init__(
        self, in_features: int | None = None, *, momentum: float = 0.9, auto_update: bool = True
    ) -> None:
        super().__init__()
        self._center: Tensor | None
        if in_features is not None:
            self._initialize(in_features=in_features)
        self.momentum = momentum
        self.auto_update = auto_update

    def _initialize(self, in_features: int) -> None:
        self.register_buffer("_center", torch.zeros(1, in_features))

    @property
    def center(self) -> Tensor:
        if self._center is None:
            raise AttributeError(f"{__class__.__name__}.center has not yet been initialized.")
        return self._center

    @implements(nn.Module)
    def forward(self, teacher_output: Tensor) -> Tensor:
        if self._center is None:
            self._initialize(in_features=teacher_output.size(1))
        centered = teacher_output - self.center
        if self.auto_update:
            self.update_center(teacher_output)
        return centered

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        """
        Update center used for teacher output.
        """
        if self._center is None:
            self._initialize(in_features=teacher_output.size(1))
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        # ema update
        self._center = self.center * self.momentum + batch_center * (1 - self.momentum)
