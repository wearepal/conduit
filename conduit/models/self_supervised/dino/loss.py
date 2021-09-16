from typing import Optional

from kit.decorators import implements
import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["DINOLoss", "EMACenter"]


class EMACenter(nn.Module):
    def __init__(
        self, in_features: Optional[int] = None, *, momentum: float = 0.9, auto_update: bool = True
    ) -> None:
        super().__init__()
        self._center: Optional[Tensor]
        self.register_buffer("_center", None)
        if in_features is not None:
            self._initialize(in_features=in_features)
        self.momentum = momentum
        self.auto_update = auto_update

    def _initialize(self, in_features: int, device: Optional[torch.device] = None) -> None:
        self._center = torch.zeros(1, in_features)
        if device is not None:
            self._center.to(device)

    @property
    def center(self) -> Tensor:
        if self._center is None:
            raise AttributeError(f"{self.__class__.__name__}.center has not yet been initialized.")
        return self._center

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        """
        Update center used for teacher output.
        """
        if self._center is None:
            self._initialize(
                in_features=teacher_output.size(1),
                device=teacher_output.device,
            )
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        # ema update
        self._center = self.center * self.momentum + batch_center * (1 - self.momentum)

    @implements(nn.Module)
    def forward(self, teacher_output: Tensor) -> Tensor:
        if self._center is None:
            self._initialize(
                in_features=teacher_output.size(1),
                device=teacher_output.device,
            )
        centered = teacher_output - self.center
        if self.auto_update:
            self.update_center(teacher_output)
        return centered


class DINOLoss(nn.Module):
    def __init__(
        self,
        *,
        student_temp: float,
        teacher_temp: float,
        warmup_teacher_temp: Optional[float] = None,
        warmup_teacher_temp_iters: int = 0,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.warmup_teacher_temp_iters = warmup_teacher_temp_iters
        self._warmup_step_size: float
        if (warmup_teacher_temp is not None) and (warmup_teacher_temp_iters > 0):
            self._warmup_step_size = (
                teacher_temp - warmup_teacher_temp
            ) / warmup_teacher_temp_iters

        self.center_momentum = center_momentum
        self.center = EMACenter(momentum=self.center_momentum)

    @implements(nn.Module)
    def forward(
        self, *, student_logits: Tensor, teacher_logits: Tensor, num_local_crops: int, step: int
    ) -> Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_logits = student_logits / self.student_temp
        student_logits_seq = student_logits.chunk(chunks=num_local_crops + 2, dim=0)

        # teacher sharpening
        if (self.warmup_teacher_temp is not None) and (step < self.warmup_teacher_temp_iters):
            teacher_temp = self.warmup_teacher_temp + (step * self._warmup_step_size)
        else:
            teacher_temp = self.teacher_temp

        teacher_probs = ((self.center(teacher_logits)) / teacher_temp).softmax(dim=-1).detach()
        teacher_probs_seq = teacher_probs.detach().chunk(chunks=2, dim=0)

        total_loss = student_logits_seq[-1].new_zeros(())
        n_loss_terms = student_logits_seq[-1].new_zeros(())
        for iq, q in enumerate(teacher_probs_seq):
            for v, sl in enumerate(student_logits_seq):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(sl, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
