import pytorch_lightning as pl

from conduit.data.datamodules.vision.cmnist import ColoredMNISTDataModule
from conduit.models.self_supervised import MoCoV2

if __name__ == "__main__":
    dm = ColoredMNISTDataModule(root="data")
    dm.setup()
    moco = MoCoV2(eval_epochs=1)
    moco.run(
        datamodule=dm,
        trainer=pl.Trainer(gpus=1, limit_train_batches=1, limit_val_batches=1),
        copy=False,
    )
