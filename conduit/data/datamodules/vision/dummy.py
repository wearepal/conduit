import attr
from ranzen import implements

from conduit.data import CdtDataModule, CdtVisionDataModule, TrainValTestSplit
from conduit.data.datasets.vision.dummy import DummyVisionDataset


@attr.define(kw_only=True)
class DummyVisionDataModule(CdtVisionDataModule):
    num_samples: int = 1_000
    seed: int = 8
    root: str = ""

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        # Split the data randomly according to val- and test-prop
        data = DummyVisionDataset(shape=(3, 32, 32), num_samples=self.num_samples, batch_size=20)
        val_data, test_data, train_data = data.random_split(props=(self.val_prop, self.test_prop))
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
