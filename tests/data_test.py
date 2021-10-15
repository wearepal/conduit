from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio.transforms as AT
from torchvision import transforms as T
from typing_extensions import Type

from conduit.data import (
    NICO,
    SSRP,
    BinarySample,
    BinarySampleIW,
    CelebA,
    NamedSample,
    SampleBase,
    TernarySample,
    TernarySampleIW,
    Waterbirds,
)
from conduit.data.datamodules import EcoacousticsDataModule
from conduit.data.datamodules.tabular.dummy import DummyTabularDataModule
from conduit.data.datamodules.vision.dummy import DummyVisionDataModule
from conduit.data.datasets import ISIC, ColoredMNIST, Ecoacoustics
from conduit.data.datasets.vision.cmnist import MNISTColorizer


@pytest.mark.parametrize("greyscale", [True, False])
@pytest.mark.parametrize("black", [True, False])
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("binarize", [True, False])
@pytest.mark.parametrize("num_channels", [1, 3, None])
def test_colorizer(
    num_channels: Optional[int], binarize: bool, background: bool, black: bool, greyscale: bool
):
    """Test label dependent transforms."""
    image_shape = [4, 7, 7]
    if num_channels is not None:
        image_shape.insert(1, num_channels)
    images = torch.rand(*image_shape)
    labels = torch.randint(low=0, high=10, size=(len(images),))
    transform = MNISTColorizer(
        scale=0.02,
        min_val=0.0,
        max_val=1.0,
        binarize=binarize,
        background=background,
        black=black,
        seed=47,
        greyscale=greyscale,
    )

    images = transform(images=images, labels=labels)


@pytest.mark.slow
@pytest.mark.parametrize("ds_cls", [ColoredMNIST, ISIC, CelebA, NICO, SSRP, Waterbirds])
def test_vision_datasets(
    root: Path,
    ds_cls: Union[
        Type[ColoredMNIST], Type[ISIC], Type[CelebA], Type[NICO], Type[SSRP], Type[Waterbirds]
    ],
) -> None:
    """Basic test for datasets.
    Confirms that the datasets can be instantiated and have a functional __getitem__ method.
    """
    transform = T.ToTensor()
    ds = ds_cls(root=root, transform=transform)
    for _ds in ds:
        assert isinstance(_ds, SampleBase)
        assert _ds.x[0] is not None
        break


@pytest.mark.slow
@pytest.mark.parametrize("ds_cls", [Ecoacoustics])
def test_audio_dataset(root: Path, ds_cls: Type[Ecoacoustics]) -> None:
    base_dir = root / "Ecoacoustics"
    target_attribute = "habitat"
    waveform_length = 60.0  # Length in seconds.
    specgram_segment_len = 30.0  # Length in seconds.

    ds = ds_cls(
        root=root,
        download=True,
        target_attr=target_attribute,
        specgram_segment_len=specgram_segment_len,
        preprocessing_transform=AT.Spectrogram(n_fft=120, hop_length=60),
        transform=None,
    )

    # Test __str__
    assert str(ds).splitlines()[0] == f"Dataset {ds.__class__.__name__}"

    # Test processed folder
    processed_audio_dir = base_dir / "Spectrogram"
    assert processed_audio_dir.exists()

    # Test __len__
    num_processed_files = len(list(processed_audio_dir.glob("**/*.pt")))
    assert len(ds) == num_processed_files

    # Test correct number of spectrogram segments are produced.
    segments_per_waveform = int(waveform_length / specgram_segment_len)
    expected_num_processed_files = len(list(base_dir.glob("**/*.wav"))) * segments_per_waveform
    assert num_processed_files == expected_num_processed_files


@pytest.mark.slow
def test_ecouacoustics_labels(root: Path):
    target_attr = "habitat"
    ds = Ecoacoustics(
        root=root,
        download=True,
        target_attr=target_attr,
        specgram_segment_len=30.0,
        preprocessing_transform=AT.Spectrogram(n_fft=120, hop_length=60),
        transform=None,
    )
    metadata = pd.read_csv(root / ds.__class__.__name__ / ds.METADATA_FILENAME)
    # Test metadata aligns with labels file.
    audio_samples_to_check = [
        "FS-08_0_20150802_0625=0.pt",
        "PL-10_0_20150604_0445=0.pt",
        "KNEPP-02_0_20150510_0730=0.pt",
    ]
    habitat_target_attributes = ["EC2", "UK1", np.nan]
    for sample, label in zip(audio_samples_to_check, habitat_target_attributes):
        matched_row = metadata.loc[metadata['fileName_pt'] == sample]
        if type(label) == str:
            assert matched_row.iloc[0][target_attr] == label
        # else:
        #     assert np.isnan(matched_row.iloc[0][target_attr])


@pytest.mark.slow
def test_ecoacoustics_dm(root: Path):
    dm = EcoacousticsDataModule(
        root=root,
        specgram_segment_len=30.0,
        preprocessing_transform=AT.Spectrogram(n_fft=120, hop_length=60),
    )
    dm.prepare_data()
    dm.setup()

    # Test loading a sample.
    train_dl = dm.train_dataloader()
    test_sample = next(iter(train_dl))

    # Test size().
    assert test_sample.x.size()[1] == dm.size().C
    assert test_sample.x.size()[2] == dm.size().H
    assert test_sample.x.size()[3] == dm.size().W


def test_add_field() -> None:
    x = torch.rand(3, 2)
    s = torch.randint(0, 2, (1, 2))
    y = torch.randint(0, 2, (1, 2))
    iw = torch.rand(1, 2)
    ns = NamedSample(x)
    assert isinstance(ns.add_field(), NamedSample)
    assert isinstance(ns.add_field(y=y), BinarySample)
    assert isinstance(ns.add_field(y=y, iw=iw), BinarySampleIW)
    assert isinstance(ns.add_field(y=y, s=s), TernarySample)
    assert isinstance(ns.add_field(y=y, s=s, iw=iw), TernarySampleIW)
    bs = BinarySample(x=x, y=y)
    assert isinstance(bs.add_field(), BinarySample)
    assert isinstance(bs.add_field(iw=iw), BinarySampleIW)
    assert isinstance(bs.add_field(s=s), TernarySample)
    assert isinstance(bs.add_field(s=s, iw=iw), TernarySampleIW)
    bsi = BinarySampleIW(x=x, y=y, iw=iw)
    assert isinstance(bsi.add_field(), BinarySampleIW)
    assert isinstance(bsi.add_field(s=s), TernarySampleIW)
    ts = TernarySample(x=x, s=s, y=y)
    assert isinstance(ts.add_field(), TernarySample)
    assert isinstance(ts.add_field(iw=iw), TernarySampleIW)
    tsi = TernarySampleIW(x=x, s=s, y=y, iw=iw)
    assert isinstance(tsi.add_field(), TernarySampleIW)


def test_tabular_dummy_data():
    dm = DummyTabularDataModule(num_samples=100, num_disc_features=1, num_cont_features=1)
    dm.prepare_data()
    dm.setup()
    train_sample = next(iter(dm.train_dataloader()))
    assert train_sample.y.shape == (60,)
    assert train_sample.x.shape == (60, 8)
    val_sample = next(iter(dm.val_dataloader()))
    assert val_sample.y.shape == (20,)
    assert val_sample.x.shape == (20, 8)
    test_sample = next(iter(dm.test_dataloader()))
    assert test_sample.y.shape == (20,)
    assert test_sample.x.shape == (20, 8)


def test_vision_dummy_data():
    dm = DummyVisionDataModule()
    dm.prepare_data()
    dm.setup()
    next(iter(dm.train_dataloader()))
