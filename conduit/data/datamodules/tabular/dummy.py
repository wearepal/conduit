import attr
from ranzen import implements

from conduit.data import CdtDataModule, TrainValTestSplit
from conduit.data.datasets.tabular.dummy import RandomTabularDataset


@attr.define(kw_only=True)
class DummyTabularDataModule(CdtDataModule):
    num_samples: int
    num_disc_features: int
    num_cont_features: int
    seed: int = 8

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        # Split the data randomly according to val- and test-prop
        data = RandomTabularDataset(
            num_cont_features=self.num_cont_features,
            num_disc_features=self.num_disc_features,
            num_samples=self.num_samples,
            seed=self.seed,
        )
        val_data, test_data, train_data = data.random_split(props=(self.val_prop, self.test_prop))
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
