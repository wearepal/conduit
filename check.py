import pytorch_lightning as pl

from conduit.data.datamodules.vision.cmnist import ColoredMNISTDataModule
from conduit.models.self_supervised import MoCoV2

dm = ColoredMNISTDataModule(root="data")
dm.setup()
moco = MoCoV2()
moco.run(datamodule=dm, trainer=pl.Trainer(), copy=False)
