import pytorch_lightning as pl
from torch import nn

from bolts.callbacks import PostHocEval
from bolts.fair.models import ErmBaseline
from tests.fair.model_test import DummyDataModule, Encoder


def test_post_hoc_eval() -> None:
    """Test the post hoc eval callback."""
    trainer = pl.Trainer(max_steps=1)
    enc = Encoder(input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128)
    clf = nn.Sequential(nn.Flatten(), nn.Linear(128, 2))
    dm = DummyDataModule()
    model = ErmBaseline(enc=enc, clf=clf, weight_decay=1e-8, lr=1e-3, lr_gamma=0.999)
    model.eval_classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 2))
    model.encoder = model.enc
    model.clf_epochs = 1
    model.datamodule = dm
    model.batch_size_eval = 10
    trainer.callbacks += [PostHocEval()]
    trainer.fit(model, datamodule=DummyDataModule())
    trainer.test(model=model, datamodule=DummyDataModule())
