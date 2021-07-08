"""CelebA DataModule."""
from __future__ import annotations
from functools import lru_cache
from typing import Any, Optional

import ethicml as em
import ethicml.vision as emvi
from kit import implements, parsable
from kit.torch import prop_random_split
from pytorch_lightning import LightningDataModule
from torchvision import transforms as T

from bolts.common import Stage
from bolts.fair.data.datasets import TiWrapper

from .base import VisionDataModule

__all__ = ["CelebaDataModule"]


class CelebaDataModule(VisionDataModule):
    """Fairness-oriented CelebA data-module."""

    @parsable
    def __init__(
        self,
        data_dir: Optional[str] = None,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
        test_split: float = 0.2,
        y_label: str = "Smiling",
        s_label: str = "Male",
        seed: int = 0,
        persist_workers: bool = False,
        cache_data: bool = False,
        pin_memory: bool = True,
        stratified_sampling: bool = False,
        sample_with_replacement: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            s_dim=1,
            y_dim=1,
            seed=seed,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            stratified_sampling=stratified_sampling,
            sample_with_replacement=sample_with_replacement,
        )
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.num_classes = 2
        self.num_sens = 2
        self.y_label = y_label
        self.s_label = s_label
        self.cache_data = cache_data

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        _, _ = em.celeba(
            download_dir=self.data_dir,
            label=self.y_label,
            sens_attr=self.s_label,
            download=True,
            check_integrity=True,
        )

    @implements(LightningDataModule)
    def setup(self, stage: Stage | None = None) -> None:
        dataset, base_dir = em.celeba(
            download_dir=self.data_dir,
            label=self.y_label,
            sens_attr=self.s_label,
            download=False,
            check_integrity=True,
        )

        tform_ls = [T.Resize(self.image_size), T.CenterCrop(self.image_size)]
        tform_ls.append(T.ToTensor())
        tform_ls.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = T.Compose(tform_ls)

        assert dataset is not None
        all_data = TiWrapper(
            emvi.TorchImageDataset(
                data=dataset.load(), root=base_dir, transform=transform, target_transform=None
            )
        )

        if self.cache_data:
            all_data.__getitem__ = lru_cache(None)(all_data.__getitem__)  # type: ignore[assignment]

        self._val_data, self._test_data, self._train_data = prop_random_split(
            dataset=all_data, props=(self.val_pcnt, self.test_pcnt), seed=self.seed
        )
