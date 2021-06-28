"""CelebA DataModule."""
from functools import lru_cache
from typing import Any, Optional

import ethicml as em
import ethicml.vision as emvi
import torch
from kit import implements
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import random_split
from torchvision import transforms as TF

from fair_bolts.datamodules.vision_datamodule import VisionBaseDataModule
from fair_bolts.datamodules.wrappers import TiWrapper


class CelebaDataModule(VisionBaseDataModule):
    """CelebA Dataset."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        image_size: int = 64,
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
    ):
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
    def setup(self, stage: Optional[str] = None) -> None:
        dataset, base_dir = em.celeba(
            download_dir=self.data_dir,
            label=self.y_label,
            sens_attr=self.s_label,
            download=False,
            check_integrity=True,
        )

        tform_ls = [TF.Resize(self.image_size), TF.CenterCrop(self.image_size)]
        tform_ls.append(TF.ToTensor())
        tform_ls.append(TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = TF.Compose(tform_ls)

        assert dataset is not None
        all_data = TiWrapper(
            emvi.TorchImageDataset(
                data=dataset.load(), root=base_dir, transform=transform, target_transform=None
            )
        )

        if self.cache_data:
            all_data.__getitem__ = lru_cache(None)(all_data.__getitem__)  # type: ignore[assignment]

        num_train_val, num_test = self._get_splits(int(len(all_data)), self.test_split)
        num_train, num_val = self._get_splits(num_train_val, self.val_split)

        g_cpu = torch.Generator()
        g_cpu = g_cpu.manual_seed(self.seed)
        self._train_data, self._val_data, self._test_data = random_split(
            all_data,
            lengths=(
                num_train,
                num_val,
                len(all_data) - num_train - num_val,
            ),
            generator=g_cpu,
        )
