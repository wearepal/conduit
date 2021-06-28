from collections import Sequence

from kit import gcopy
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader


class PostHocEval(pl.Callback):
    """
    Trains and evaluates an auxillary classifier model.
    Your model should have:
        - ``self.classifier``
        - ``self.datamodule``
        - ``self.clf_epochs``
        - ``self.trainer''
    """

    def __init__(self) -> None:
        super().__init__()

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.classifier = pl_module.classifier.to(pl_module.device)
        pl_module.eval_trainer = gcopy(trainer, max_epochs=pl_module.clf_epochs)

    def _eval_loop(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
        train_dl: DataLoader,
        test_dl: DataLoader,
    ) -> None:
        trainer.fit(model, train_dataloader=train_dl)
        trainer.test(dataloaders=test_dl)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._eval_loop(
            trainer=pl_module.eval_trainer,
            model=pl_module.classifier,
            train_dl=pl_module.datamodule.train_dataloader(
                eval=True, batch_size=pl_module.batch_size_eval
            ),
            test_dl=pl_module.datamodule.val_dataloader(),
        )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._eval_loop(
            trainer=pl_module.eval_trainer,
            model=pl_module.classifier,
            train_dl=pl_module.datamodule.train_dataloader(
                eval=True, batch_size=pl_module.batch_size_eval
            ),
            test_dl=pl_module.datamodule.val_dataloader(),
        )

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._eval_loop(
            trainer=pl_module.eval_trainer,
            model=pl_module.classifier,
            train_dl=pl_module.datamodule.train_dataloader(
                eval=True, batch_size=pl_module.batch_size_eval
            ),
            test_dl=pl_module.datamodule.test_dataloader(),
        )
