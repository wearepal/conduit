from typing import Any

import pytorch_lightning as pl
from ranzen.decorators import implements

from conduit.models.self_supervised.dino.utils import cosine_scheduler

__all__ = ["DINOScheduler"]


class DINOScheduler(pl.Callback):
    def __init__(
        self,
        base_lr: float,
        min_lr: float,
        total_iters: int,
        base_wd: float,
        min_wd: float,
        warmup_iters: int = 0,
        start_warmup_lr: float = 0,
    ) -> None:
        super().__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_iters = total_iters
        self.base_wd = base_wd
        self.min_wd = min_wd
        self.warmup_iters = warmup_iters
        self.start_warmup_lr = start_warmup_lr

        self._lr_schedule = cosine_scheduler(
            base_value=base_lr,
            final_value=min_lr,
            total_iters=max_steps,  # type: ignore
            warmup_iters=min(total_iters - 1, warmup_iters),
            start_warmup_value=start_warmup_lr,
        )
        self._wd_schedule = cosine_scheduler(
            base_value=base_wd,
            final_value=min_wd,
            total_iters=total_iters,
        )

    @implements(pl.Callback)
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        assert self._trainer.optimizers is not None  # type: ignore
        for i, param_group in enumerate(self._trainer.optimizers[0].param_groups):  # type: ignore
            param_group["lr"] = self._lr_schedule[batch_idx]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self._wd_schedule[batch_idx]
