import time

import pytest
import pytorch_lightning as pl
from ranzen.torch.data import TrainingMode
from ranzen.torch.loss import cross_entropy_loss
import torch
from torch import Tensor
import torch.nn as nn

from conduit.data.datamodules.vision.dummy import DummyVisionDataModule
from conduit.data.structures import TernarySample
from conduit.progress import CdtProgressBar


class DummyModel(pl.LightningModule):
    def __init__(self, feature_dim: int, *, out_dim: int, wait_time: float = 0.0) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((feature_dim, feature_dim)),
            nn.Flatten(),
            nn.Linear(feature_dim**2 * 3, out_dim),
        )
        self.wait_time = wait_time

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters())

    def training_step(self, sample: TernarySample[Tensor], *args) -> Tensor:  # type: ignore
        logits = self.model(sample.x)
        time.sleep(self.wait_time)
        return cross_entropy_loss(logits, target=sample.y)

    def validation_step(self, sample: TernarySample[Tensor], *args) -> Tensor:  # type: ignore
        logits = self.model(sample.x)
        time.sleep(self.wait_time)
        return cross_entropy_loss(logits, target=sample.y)

    def test_step(self, sample: TernarySample[Tensor], *args) -> Tensor:  # type: ignore
        logits = self.model(sample.x)
        time.sleep(self.wait_time)
        return cross_entropy_loss(logits, target=sample.y)

    def predict_step(self, sample: TernarySample[Tensor], *args) -> Tensor:  # type: ignore
        time.sleep(self.wait_time)
        return self.model(sample.x).argmax(dim=1)


@pytest.mark.parametrize("training_mode", list(TrainingMode))
@pytest.mark.parametrize("theme", list(CdtProgressBar.Theme))
def test_cdt_progbar(training_mode: TrainingMode, theme: CdtProgressBar.Theme):
    dm = DummyVisionDataModule(num_workers=4, training_mode=training_mode)
    assert isinstance(dm.y_card, int)
    model = DummyModel(feature_dim=1, out_dim=dm.y_card, wait_time=0.0)
    trainer = pl.Trainer(
        max_steps=2,
        val_check_interval=1,
        log_every_n_steps=1,
        callbacks=[CdtProgressBar(theme=theme)],
        num_sanity_val_steps=1,
    )
    dm.setup()
    trainer.fit(
        model=model,
        datamodule=dm,
    )
    trainer.test(
        model=model,
        datamodule=dm,
    )
    trainer.predict(
        model=model,
        dataloaders=[dm.test_dataloader()],
    )
