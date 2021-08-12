from __future__ import annotations
from pathlib import Path

from bolts.data.datasets.base import PBDataset
from bolts.data.datasets.utils import AudioLoadingBackend, AudioTform, infer_al_backend


class PBAudioDataset(PBDataset):
    """Base dataset for audio data."""

    def __init__(
        self, *, x, audio_dir: Path | str, y=None, s=None, transform: AudioTform | None = None
    ) -> None:
        super().__init__(x=x, y=y, s=s)

        # Convert string path to Path object.
        if isinstance(audio_dir, str):
            audio_dir = Path(audio_dir)

        self.audio_dir = audio_dir
        self.transform = transform

        # Infer the appropriate audio-loading backend based on the operating system.
        self.al_backend: AudioLoadingBackend = infer_al_backend()
        self.log(f'Using {self.al_backend} as backend for audio-loading')
