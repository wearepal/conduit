"""K&C Baseline model."""
from ranzen import implements
from ranzen.torch import CrossEntropyLoss, ReductionType, TrainingMode
from torch import Tensor, nn

from conduit.data.structures import TernarySampleIW
from conduit.fair.models.erm import ERMClassifierF
from conduit.models.erm import ERMClassifier

__all__ = ["KC"]


class KC(ERMClassifierF):
    """Kamiran and Calders' instance-weighting method."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        clf: nn.Module,
        lr: float = 3.0e-4,
        weight_decay: float = 0.0,
        lr_initial_restart: int = 10,
        lr_restart_mult: int = 2,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
    ) -> None:
        super().__init__(
            encoder=encoder,
            clf=clf,
            lr=lr,
            weight_decay=weight_decay,
            lr_initial_restart=lr_initial_restart,
            lr_restart_mult=lr_restart_mult,
            lr_sched_interval=lr_sched_interval,
            lr_sched_freq=lr_sched_freq,
            loss_fn=CrossEntropyLoss(reduction=ReductionType.mean),
        )

    @implements(ERMClassifier)
    def _get_loss(self, logits: Tensor, *, batch: TernarySampleIW) -> Tensor:
        return self.loss_fn(input=logits, target=batch.y, instance_weight=batch.iw)
