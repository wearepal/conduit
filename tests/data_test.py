from pathlib import Path
from typing import Optional, Tuple, Union

from albumentations.pytorch import ToTensorV2
import numpy as np
import pytest
import torch
from torch import Tensor, nn
import torchaudio.transforms as AT
from torchvision import transforms as T
from typing_extensions import Type

from conduit.data import (
    NICO,
    SSRP,
    BinarySample,
    BinarySampleIW,
    CdtVisionDataModule,
    CelebA,
    CelebADataModule,
    ColoredMNISTDataModule,
    ImageTform,
    NamedSample,
    NICODataModule,
    SampleBase,
    SoundscapeAttr,
    TernarySample,
    TernarySampleIW,
    Waterbirds,
    WaterbirdsDataModule,
)
from conduit.data.datamodules import EcoacousticsDataModule
from conduit.data.datamodules.tabular.dummy import DummyTabularDataModule
from conduit.data.datamodules.vision.dummy import DummyVisionDataModule
from conduit.data.datasets import ISIC, ColoredMNIST, Ecoacoustics
from conduit.data.datasets.utils import get_group_ids, stratified_split
from conduit.data.datasets.vision.cmnist import MNISTColorizer
from conduit.fair.data.datasets.dummy import DummyDataset
from conduit.transforms import Framing, LogMelSpectrogram, LogMelSpectrogramNp
from conduit.transforms.audio import FramingNp


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
@pytest.mark.parametrize(
    "dm", [ColoredMNISTDataModule, CelebADataModule, NICODataModule, WaterbirdsDataModule]
)
def test_vision_datamodules(root, dm: Type[CdtVisionDataModule]):
    dm = dm(root=root)
    dm.prepare_data()
    dm.setup()
    _ = next(iter(dm.train_dataloader()))


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
@pytest.mark.parametrize("segment_len", [1, 15, 30])
def test_audio_dataset(root: Path, segment_len: float) -> None:

    ds = Ecoacoustics(
        root=str(root),
        download=True,
        target_attrs=[SoundscapeAttr.habitat, SoundscapeAttr.site],
        transform=None,
        segment_len=segment_len,
    )

    assert len(ds) == len(ds.x) == len(ds.metadata)
    assert ds.y is not None
    assert ds.y.shape == (len(ds), 2)
    assert ds.y.dtype == torch.long
    num_frames = (
        ds._MAX_AUDIO_LEN * ds.sample_rate if segment_len is None else segment_len * ds.sample_rate
    )
    for idx in (0, -1):
        sample = ds[idx]
        assert isinstance(sample, BinarySample)
        assert isinstance(sample.x, Tensor)
        assert sample.x.shape == (1, num_frames)


@pytest.mark.slow
def test_ecoacoustics_labels(root: Path):
    target_attr = [SoundscapeAttr.habitat]
    ds = Ecoacoustics(
        root=str(root),
        download=True,
        target_attrs=target_attr,
        segment_len=1,
        transform=None,
    )
    # Test metadata aligns with labels file.
    audio_samples_to_check = [
        "FS-08_0_20150802_0625.wav",
        "PL-10_0_20150604_0445.wav",
        "KNEPP-02_0_20150510_0730.wav",
    ]
    habitat_target_attributes = ["EC2", "UK1", np.nan]
    for sample, label in zip(audio_samples_to_check, habitat_target_attributes):
        matched_row = ds.metadata.loc[ds.metadata["fileName"] == sample]
        if isinstance(label, str):
            assert matched_row.iloc[0][str(target_attr[0])] == label


@pytest.mark.slow
def test_ecoacoustics_dm(root: Path):
    dm = EcoacousticsDataModule(
        root=str(root),
        segment_len=30.0,
        target_attrs=[SoundscapeAttr.habitat],
        train_transforms=AT.Spectrogram(),
    )
    dm.prepare_data()
    dm.setup()

    # Test loading a sample.
    train_dl = dm.train_dataloader()
    test_sample = next(iter(train_dl))

    # Test size().
    assert test_sample.x.size()[1] == dm.dims[0]
    assert test_sample.x.size()[2] == dm.dims[1]
    assert test_sample.x.size()[3] == dm.dims[2]


@pytest.mark.slow
def test_spectrogram_transform(root: Path):
    dm = EcoacousticsDataModule(
        root=str(root),
        segment_len=30.0,
        target_attrs=[SoundscapeAttr.habitat],
        train_transforms=T.Compose([nn.Identity()]),
        test_transforms=T.Compose([nn.Identity()]),
        train_batch_size=1,
    )
    dm.prepare_data()
    dm.setup()

    # Test loading a sample.
    train_dl = dm.train_dataloader()
    sample = next(iter(train_dl))

    lms = LogMelSpectrogram(
        sample_rate=16_000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        f_min=125.0,
        f_max=7_500.0,
        pad=0,
        n_mels=64,
        window_fn=None,
        power=2.0,
        normalized=False,
        wkwargs=None,
        center=True,
        pad_mode="reflect",
        onesided=True,
        norm=None,
        mel_scale="htk",
        log_offset=0.01,
    )

    lmsnp = LogMelSpectrogramNp(
        sample_rate=16_000,
        stft_window_length_seconds=0.025,
        stft_hop_length_seconds=0.010,
        log_offset=0.01,  # Offset used for stabilized log of input mel-spectrogram.
        num_mel_bins=64,  # Frequency bands in input mel-spectrogram patch.
        mel_min_hz=125,
        mel_max_hz=7_500,
        mel_break_freq_hertz=700.0,
        mel_high_freq_q=1_127.0,
    )

    a = lms(sample.x.clone())
    b = lmsnp(sample.x.clone())

    fr = Framing()
    frnp = FramingNp()

    c = fr(a)
    d = frnp(b)

    assert c.shape == d.shape


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


@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("disc_feats", [1, 100])
@pytest.mark.parametrize("cont_feats", [1, 10])
@pytest.mark.parametrize("s_card", [None, 2, 10])
@pytest.mark.parametrize("y_card", [None, 2, 10])
def test_tabular_dummy_data(
    batch_size: int, disc_feats: int, cont_feats: int, s_card: Optional[int], y_card: Optional[int]
):
    dm = DummyTabularDataModule(
        num_samples=1_000,
        num_disc_features=disc_feats,
        num_cont_features=cont_feats,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        s_card=s_card,
        y_card=y_card,
    )
    dm.prepare_data()
    dm.setup()
    for loader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        sample = next(iter(loader))
        if s_card is None:
            assert isinstance(sample, (NamedSample, BinarySample))
        else:
            assert sample.s.shape == (batch_size,)
        if y_card is None:
            assert isinstance(sample, (NamedSample))
        else:
            assert sample.y.shape == (batch_size,)
        for group in dm.train_data.feature_groups:
            assert sample.x[:, group].sum() == batch_size
        assert sample.x[:, dm.train_data.cont_indexes].shape == (batch_size, cont_feats)


@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("size", [8, 32])
@pytest.mark.parametrize("s_card", [None, 10])
@pytest.mark.parametrize("y_card", [None, 10])
@pytest.mark.parametrize("channels_transforms", [(1, ToTensorV2()), (3, None)])
def test_vision_dummy_data(
    batch_size: int,
    size: int,
    s_card: Optional[int],
    y_card: Optional[int],
    channels_transforms: Tuple[int, ImageTform],
):
    channels, transforms = channels_transforms
    dm = DummyVisionDataModule(
        train_batch_size=batch_size,
        eval_batch_size=batch_size,  # type: ignore
        height=size,
        width=size,
        s_card=s_card,
        y_card=y_card,
        channels=channels,
        train_transforms=transforms,  # type: ignore
        test_transforms=transforms,  # type: ignore
    )
    dm.prepare_data()
    dm.setup()
    for loader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        sample = next(iter(loader))
        assert sample.x.shape == (batch_size, channels, size, size)
        if s_card is None:
            assert isinstance(sample, (NamedSample, BinarySample))
        else:
            assert sample.s.shape == (batch_size,)
        if y_card is None:
            assert isinstance(sample, (NamedSample))
        else:
            assert isinstance(sample, (BinarySample))
            assert sample.y.shape == (batch_size,)


def test_stratified_split():
    ds = DummyDataset((3, 28, 28), (1,), (1,), (1,), num_samples=117, seed=47)
    train, test = stratified_split(dataset=ds, default_train_prop=0.5)
    ids_train = get_group_ids(train)
    ids_test = get_group_ids(test)
    counts_train = ids_train.unique(return_counts=True)[1]
    counts_test = ids_test.unique(return_counts=True)[1]
    assert torch.isclose(counts_train, counts_test, atol=1).all()

    train_props = {0: {0: 0.25, 1: 0.45}, 1: 0.3}
    train, test = stratified_split(
        dataset=ds,
        default_train_prop=0.5,
        train_props=train_props,
    )
    mask_train = train.y == 1
    mask_test = test.y == 1
    mask_all = ds.y == 1

    n_train = mask_train.count_nonzero().item()
    n_test = mask_test.count_nonzero().item()
    n_all = mask_all.count_nonzero().item()

    assert n_train == pytest.approx(0.30 * n_all, abs=1)
    assert n_test == pytest.approx(0.70 * n_all, abs=1)

    mask_train = (train.s == 0) & (train.y == 0)
    mask_test = (test.s == 0) & (test.y == 0)
    mask_all = (ds.s == 0) & (ds.y == 0)

    n_train = mask_train.count_nonzero().item()
    n_test = mask_test.count_nonzero().item()
    n_all = mask_all.count_nonzero().item()

    assert n_train == pytest.approx(0.25 * n_all, abs=1)
    assert n_test == pytest.approx(0.75 * n_all, abs=1)

    mask_train = (train.s == 1) & (train.y == 0)
    mask_test = (test.s == 1) & (test.y == 0)
    mask_all = (ds.s == 1) & (ds.y == 0)

    n_train = mask_train.count_nonzero().item()
    n_test = mask_test.count_nonzero().item()
    n_all = mask_all.count_nonzero().item()

    assert n_train == pytest.approx(0.45 * n_all, abs=1)
    assert n_test == pytest.approx(0.55 * n_all, abs=1)
