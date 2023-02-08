"""Base class for audio datasets."""
from typing import Optional, TypeVar, final

import attr
import torchaudio.transforms as T  # type: ignore
from typing_extensions import override

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.base import I
from conduit.data.datasets.utils import AudioTform
from conduit.data.datasets.wrappers import AudioTransformer, InstanceWeightedDataset
from conduit.data.structures import SizedDataset
from conduit.types import Stage

__all__ = ["CdtAudioDataModule"]

D = TypeVar("D", bound=SizedDataset)


@attr.define(kw_only=True)
class CdtAudioDataModule(CdtDataModule[D, I]):
    root: str = attr.field(kw_only=False)
    _train_transforms: Optional[AudioTform] = None
    _test_transforms: Optional[AudioTform] = None

    @property
    @final
    def train_transforms(self) -> AudioTform:
        return (
            self._default_train_transforms
            if self._train_transforms is None
            else self._train_transforms
        )

    @train_transforms.setter
    def train_transforms(self, transform: Optional[AudioTform]) -> None:
        self._train_transforms = transform
        if isinstance(self._train_data, AudioTransformer):
            self._train_data.transform = transform

    @property
    @final
    def test_transforms(self) -> AudioTform:
        return (
            self._default_test_transforms
            if self._test_transforms is None
            else self._test_transforms
        )

    @test_transforms.setter
    @final
    def test_transforms(self, transform: Optional[AudioTform]) -> None:
        self._test_transforms = transform
        if isinstance(self._val_data, AudioTransformer):
            self._val_data.transform = transform
        if isinstance(self._test_data, AudioTransformer):
            self._test_data.transform = transform

    @property
    def _default_train_transforms(self) -> T.Spectrogram:
        return T.Spectrogram()

    @property
    def _default_test_transforms(self) -> T.Spectrogram:
        return T.Spectrogram()

    @override
    @final
    def _setup(self, stage: Optional[Stage] = None) -> None:
        train, val, test = self._get_splits()
        train = AudioTransformer(train, transform=self.train_transforms)
        if self.instance_weighting:
            train = InstanceWeightedDataset(train)
        self._train_data = train
        self._val_data = AudioTransformer(val, transform=self.test_transforms)
        self._test_data = AudioTransformer(test, transform=self.test_transforms)
