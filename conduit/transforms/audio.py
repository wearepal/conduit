from typing import Callable, List, Optional

import numpy as np
from ranzen import parsable
import torch
from torch import Tensor, nn
import torchaudio.transforms as T  # type: ignore

from conduit.data.datasets.utils import AudioTform

__all__ = ["LogMelSpectrogram", "Framing", "Compose"]


class LogMelSpectrogram(T.MelSpectrogram):
    @parsable
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Optional[Callable] = None,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        log_offset: float = 0.0,
    ):
        if window_fn is None:
            window_fn = torch.hann_window

        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
            norm=norm,
            mel_scale=mel_scale,
        )
        self.log_offset = log_offset

    def forward(self, waveform: Tensor) -> Tensor:
        x = super().forward(waveform)
        return torch.log(x + self.log_offset)


class Framing(nn.Module):
    def __init__(
        self,
        stft_hop_length_seconds: float = 0.010,
        example_window_seconds: float = 0.96,  # Each example contains 96 10ms frames
        example_hop_seconds: float = 0.96,  # with zero overlap.
    ):
        super().__init__()
        self.stft_hop_length_seconds = stft_hop_length_seconds
        self.example_window_seconds = example_window_seconds
        self.example_hop_seconds = example_hop_seconds

    def forward(self, data: Tensor) -> Tensor:
        # Frame features into examples.
        features_sample_rate = 1.0 / self.stft_hop_length_seconds
        window_length = int(round(self.example_window_seconds * features_sample_rate))
        hop_length = int(round(self.example_hop_seconds * features_sample_rate))

        _, _, num_samples = data.shape
        num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
        strides = (1, 1, hop_length)

        return (
            data.unsqueeze(1)
            .as_strided(size=(num_frames, 64, window_length), stride=strides)
            .unsqueeze(1)
            .permute(0, 1, 3, 2)
        )


class Compose(nn.Module):
    def __init__(self, transforms: List[AudioTform]) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, audio: Tensor) -> Tensor:
        for tform in self.transforms:
            audio = tform(audio)
        return audio
