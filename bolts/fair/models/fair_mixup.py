"""Implementation of ICLR 21 Fair Mixup."""
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


class FairMixup(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        ...

    def _inference_step(self):
        ...

    def _inference_end(self):
        ...

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        ...

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        ...

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        ...

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        ...
