"""Base class for audio datasets."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, final
from typing_extensions import override

from torch import Tensor
import torchaudio.transforms as T  # type: ignore

from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datasets.audio.base import CdtAudioDataset
from conduit.data.datasets.base import I
from conduit.data.datasets.utils import AudioTform
from conduit.data.datasets.wrappers import AudioTransformer
from conduit.data.structures import TrainValTestSplit

__all__ = ["CdtAudioDataModule"]


@dataclass(kw_only=True)
class CdtAudioDataModule(CdtDataModule[AudioTransformer, I]):
    root: str = field(kw_only=False)
    train_tf: Optional[AudioTform] = None
    test_tf: Optional[AudioTform] = None

    @property
    @final
    def train_transforms(self) -> AudioTform:
        return self._default_train_transforms if self.train_tf is None else self.train_tf

    @train_transforms.setter
    def train_transforms(self, transform: Optional[AudioTform]) -> None:
        self.train_tf = transform
        if isinstance(self._train_data, AudioTransformer):
            self._train_data.transform = transform

    @property
    @final
    def test_transforms(self) -> AudioTform:
        return self._default_test_transforms if self.test_tf is None else self.test_tf

    @test_transforms.setter
    @final
    def test_transforms(self, transform: Optional[AudioTform]) -> None:
        self.test_tf = transform
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

    @abstractmethod
    def _get_audio_splits(self) -> TrainValTestSplit[CdtAudioDataset[I, Tensor, Tensor]]:
        raise NotImplementedError()

    @override
    def _get_splits(self) -> TrainValTestSplit[AudioTransformer]:
        train, val, test = self._get_audio_splits()
        return TrainValTestSplit(
            train=AudioTransformer(train, transform=self.train_transforms),
            val=AudioTransformer(val, transform=self.test_transforms),
            test=AudioTransformer(test, transform=self.test_transforms),
        )
