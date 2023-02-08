from pathlib import Path
from typing import List, Optional, Union, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import torchaudio  # type: ignore
from typing_extensions import override

from conduit.data.datasets.base import CdtDataset, I, S, Y
from conduit.data.datasets.utils import (
    AudioLoadingBackend,
    AudioTform,
    apply_audio_transform,
    infer_al_backend,
)
from conduit.data.structures import TargetData
from conduit.types import IndexType

__all__ = ["CdtAudioDataset"]


class CdtAudioDataset(CdtDataset[I, npt.NDArray[np.string_], Y, S]):
    """Base dataset for audio data."""

    x: npt.NDArray[np.string_]

    def __init__(
        self,
        audio_dir: Union[Path, str],
        *,
        x: npt.NDArray[np.string_],
        y: Optional[TargetData] = None,
        s: Optional[TargetData] = None,
        transform: Optional[AudioTform] = None,
    ) -> None:
        super().__init__(x=x, y=y, s=s)

        # Convert string path to Path object.
        if isinstance(audio_dir, str):
            audio_dir = Path(audio_dir)

        self.audio_dir = audio_dir
        self.transform = transform

        # Infer the appropriate audio-loading backend based on the operating system.
        self.al_backend: AudioLoadingBackend = infer_al_backend()
        self.logger.info(f"Using {self.al_backend} as backend for audio-loading.")
        torchaudio.set_audio_backend(self.al_backend)

    def __repr__(self) -> str:
        head = f"Dataset {self.__class__.__name__}"
        body = [
            f"Number of datapoints: {len(self)}",
            f"Base audio-directory location: {self.audio_dir.resolve()}",
            *self.extra_repr().splitlines(),
        ]
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def load_sample(self, index: IndexType) -> Tensor:
        def _load(_filename: Path) -> Tensor:
            return torchaudio.load(self.audio_dir / _filename)[0]  # type: ignore

        if isinstance(index, int):
            return _load(self.audio_dir / self.x[index])
        return torch.cat([_load(filepath) for filepath in self.x[index]], dim=0)

    @overload
    def _sample_x(self, index: int, *, coerce_to_tensor: bool = ...) -> Tensor:
        ...

    @overload
    def _sample_x(self, index: List[int], *, coerce_to_tensor: bool = ...) -> List[Tensor]:
        ...

    @override
    def _sample_x(
        self, index: IndexType, *, coerce_to_tensor: bool = False
    ) -> Union[Tensor, List[Tensor]]:
        waveform = self.load_sample(index)
        return apply_audio_transform(waveform, transform=None)
