from collections import Sequence
from typing import NamedTuple, Optional

from kit import gcopy
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

__all__ = ["PostHocEval"]


class EvalModule(pl.LightningModule):
    def __init__(self, enc: nn.Module, clf: nn.Module):
        super().__init__()
        self.enc = enc
        self.clf = gcopy(clf)
        self._loss = nn.CrossEntropyLoss()

    def training_step(self, batch: NamedTuple, batch_idx: int) -> STEP_OUTPUT:
        y = self(batch.x)
        return self._loss(y, batch.y.view(-1).long())

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            self.enc.eval()
            z = self.enc(x)
        return self.clf(z)

    def test_step(self, batch: NamedTuple, batch_idx: int) -> Optional[STEP_OUTPUT]:
        logits = self(batch.x)
        return {
            "y": batch.y.view(-1),
            "preds": logits.sigmoid().round().squeeze(-1),
        }

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        torch.cat([_r["y"] for _r in outputs], 0)
        torch.cat([_r["preds"] for _r in outputs], 0)

    @staticmethod
    def _maybe_reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()  # type: ignore

    def reset_parameters(self):
        self.clf.apply(self._maybe_reset_parameters)

    def configure_optimizers(self):
        return AdamW(self.clf.parameters())


class PostHocEval(pl.Callback):
    """
    Trains and evaluates an auxillary classifier model.
    Your model should have:
        - ``self.classifier``
        - ``self.encoder``
        - ``self.datamodule``
        - ``self.clf_epochs``
        - ``self.trainer''
    """

    def __init__(self) -> None:
        super().__init__()

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.eval_trainer = gcopy(
            trainer,
            max_epochs=pl_module.clf_epochs,
            callbacks=[cb for cb in trainer.callbacks if not isinstance(cb, PostHocEval)],
        )

    def _eval_loop(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
        train_dl: DataLoader,
        test_dl: DataLoader,
    ) -> None:
        trainer.fit(model, train_dataloader=train_dl)
        trainer.test(test_dataloaders=test_dl)

    def _call_eval_loop(self, pl_module: pl.LightningModule):
        pl_module.reset_parameters()
        self._eval_loop(
            trainer=pl_module.eval_trainer,
            model=EvalModule(enc=pl_module.encoder, clf=pl_module.classifier),
            train_dl=pl_module.datamodule.train_dataloader(),
            test_dl=pl_module.datamodule.val_dataloader(),
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._call_eval_loop(pl_module)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._call_eval_loop(pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.reset_parameters()
        self._eval_loop(
            trainer=pl_module.eval_trainer,
            model=EvalModule(enc=pl_module.encoder, clf=pl_module.classifier),
            train_dl=pl_module.datamodule.train_dataloader(),
            test_dl=pl_module.datamodule.test_dataloader(),
        )
