from typing import List, Optional, Sequence

from kit import gcopy
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from typing_extensions import Type

from conduit.models.erm import FineTuner

__all__ = ["PostHocEval"]


class PostHocEval(pl.Callback):
    """
    Trains and evaluates an auxillary classifier model.
    Your model should have:
        - ``self.eval_predictor``
        - ``self.encoder``
        - ``self.datamodule``
        - ``self.eval_epochs``
        - ``self.trainer''
    """

    eval_clf: FineTuner

    def __init__(self, callbacks_to_ignore: Optional[List[Type[pl.Callback]]] = None) -> None:
        super().__init__()
        self.ignored_callbacks = (
            tuple(callbacks_to_ignore) if callbacks_to_ignore is not None else ()
        )

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.eval_trainer = gcopy(
            trainer,
            callbacks=[
                cb
                for cb in trainer.callbacks
                if not isinstance(cb, (PostHocEval,) + self.ignored_callbacks)
            ],
        )
        trainer.fit_loop.max_epochs = pl_module.clf_epochs
        self.eval_clf = FineTuner(encoder=pl_module.encoder, classifier=pl_module.eval_classifier)

    @staticmethod
    def _eval_loop(
        trainer: pl.Trainer,
        model: pl.LightningModule,
        train_dl: DataLoader,
        test_dl: DataLoader,
    ) -> None:
        trainer.fit(model, train_dataloader=train_dl)
        if trainer.fast_dev_run:
            trainer.test(model=model, test_dataloaders=test_dl)
        else:
            trainer.test(test_dataloaders=test_dl)

    def _call_eval_loop(self, pl_module: pl.LightningModule) -> None:
        self.eval_clf.reset_parameters()
        self._eval_loop(
            trainer=pl_module.eval_trainer,
            model=self.eval_clf,
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
        self.eval_clf.reset_parameters()
        self._eval_loop(
            trainer=pl_module.eval_trainer,
            model=self.eval_clf,
            train_dl=pl_module.datamodule.train_dataloader(),
            test_dl=pl_module.datamodule.test_dataloader(),
        )
