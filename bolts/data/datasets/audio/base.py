from __future__ import annotations
from pathlib import Path

from bolts.data.datasets.base import PBDataset
import torchaudio.transforms as AudioTform  # TODO: Add to dataset.utils


class PBAudioDataset(PBDataset):
    """Base dataset for audio data."""

    def __init__(
        self,
        *,
        x,
        audio_dir: Path | str,
        y = None,
        s = None,
        transform: AudioTform | None = None
    ) -> None:
        super().__init__(x=x, y=y, s=s)

        # Convert string path to Path object.
        if isinstance(audio_dir, str):
            audio_dir = Path(audio_dir)

        self.audio_dir = audio_dir
        self.transform = transform
