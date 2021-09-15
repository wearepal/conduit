from __future__ import annotations

from kit import implements

from conduit.data import CdtDataModule, TrainValTestSplit
from conduit.data.datasets.tabular.dummy import RandomTabularDataset


class DummyTabularDataModule(CdtDataModule):
    def __init__(
        self, num_samples: int, num_disc_features: int, num_cont_features: int, seed: int = 8
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_disc_features = num_disc_features
        self.num_cont_features = num_cont_features
        self.seed = seed

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
