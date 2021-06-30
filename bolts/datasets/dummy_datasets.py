import torch
from torch.utils.data import Dataset

from bolts.datasets.ethicml_datasets import DataBatch


class DummyDataset(Dataset):
    """Generate a dummy dataset."""

    def __init__(self, *shapes, num_samples: int = 10000):
        """
        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset
        """
        super().__init__()
        self.shapes = shapes
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        sample = []
        for shape in self.shapes:
            spl = torch.rand(*shape)
            sample.append(spl)
        return DataBatch(x=sample[0], s=sample[1].round(), y=sample[2].round(), iw=sample[3])
