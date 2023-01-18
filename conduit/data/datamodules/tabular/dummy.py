from typing import Optional

import attr
from typing_extensions import override

from conduit.data import TrainValTestSplit
from conduit.data.datamodules import CdtDataModule
from conduit.data.datasets.tabular import RandomTabularDataset


@attr.define(kw_only=True)
class DummyTabularDataModule(CdtDataModule):
    num_samples: int
    num_disc_features: int
    num_cont_features: int
    seed: int = 8
    s_card: Optional[int] = None
    y_card: Optional[int] = None

    @override
    def _get_splits(self) -> TrainValTestSplit[RandomTabularDataset]:
        # Split the data randomly according to val- and test-prop
        data = RandomTabularDataset(
            num_cont_features=self.num_cont_features,
            num_disc_features=self.num_disc_features,
            num_samples=self.num_samples,
            seed=self.seed,
            s_card=self.s_card,
            y_card=self.y_card,
        )
        val_data, test_data, train_data = data.random_split(props=(self.val_prop, self.test_prop))
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)
