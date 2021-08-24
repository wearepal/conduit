from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DINOLoss"]


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
