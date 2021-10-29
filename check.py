from dataclasses import dataclass
import pytorch_lightning as pl
import torch.nn as nn

from conduit.data.datamodules.vision.cmnist import ColoredMNISTDataModule
from conduit.models.erm import FineTuner
from conduit.models.self_supervised import DINO, MoCoV2
import pytorch_lightning as pl
import attr

@attr.define(kw_only=True, eq=False, repr=False)
# @dataclass(repr=False, eq=False)
class DummyModel(nn.Module):
    model: nn.Module = attr.field(default=nn.Linear(2, 1))

    def __attrs_post_init__(self):
        super().__init__()
        self.net = nn.Linear(2, 1)

    def __attrs_pre_init__(self):
        super().__init__()

if __name__ == "__main__":
    # dm = ColoredMNISTDataModule(root="data")
    # dm.setup()
    # moco = MoCoV2(eval_epochs=1)
    # moco.run(
    #     datamodule=dm,
    #     trainer=pl.Trainer(gpus=0, limit_train_batches=1, limit_val_batches=1),
    #     copy=False,
    # )
    model = DummyModel()
    # print(tuple(model.model.parameters()))
    print(list(model.parameters()))
    print(list(model.model.parameters()))

    # FineTuner(encoder=nn.Linear(1, 2), classifier=nn.Linear(2, 1))
