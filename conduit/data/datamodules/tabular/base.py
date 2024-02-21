from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar
from typing_extensions import override

from ranzen import some

from conduit.data.datamodules import CdtDataModule
from conduit.data.datasets.tabular import CdtTabularDataset
from conduit.data.datasets.wrappers import InstanceWeightedDataset
from conduit.data.structures import TernarySample
from conduit.transforms.tabular import TabularNormalize
from conduit.types import Stage

__all__ = ["CdtTabularDataModule"]

D = TypeVar("D", bound=CdtTabularDataset)


@dataclass(kw_only=True, eq=False)
class CdtTabularDataModule(CdtDataModule[D, TernarySample]):
    """Base for tabular data modules."""

    @abstractmethod
    def _default_transforms(self) -> TabularNormalize | None:
        """Return the default tabular transform."""
        raise NotImplementedError()

    @override
    def _setup(self, stage: Stage | None = None) -> None:
        super()._setup(stage)

        if some(tf := self._default_transforms()):
            assert not isinstance(
                self._train_data, InstanceWeightedDataset
            ), "Combination of instance weights and tfs is not yet supported for tabular data."
            assert some(self._train_data)
            assert some(self._val_data)
            assert some(self._test_data)

            self._train_data.fit_transform_(tf)
            self._val_data.transform_(tf)
            self._test_data.transform_(tf)
