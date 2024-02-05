from contextlib import suppress
from pathlib import Path
from typing import Optional, Tuple, Union
from typing_extensions import Type

from albumentations.pytorch import ToTensorV2  # type: ignore
import numpy as np
import pytest
import torch
from torch import Tensor
from torchvision import transforms as T  # type: ignore

from conduit.data import (
    BinarySample,
    BinarySampleIW,
    NamedSample,
    SampleBase,
    SubgroupSample,
    TernarySample,
    TernarySampleIW,
)
from conduit.data.datamodules.audio import EcoacousticsDataModule
from conduit.data.datamodules.tabular.dummy import DummyTabularDataModule
from conduit.data.datamodules.vision import (
    CdtVisionDataModule,
    CelebADataModule,
    ColoredMNISTDataModule,
    NICODataModule,
    WaterbirdsDataModule,
)
from conduit.data.datamodules.vision.dummy import DummyVisionDataModule
from conduit.data.datasets import get_group_ids, stratified_split
from conduit.data.datasets.audio import Ecoacoustics, SoundscapeAttr
from conduit.data.datasets.vision import (
    Camelyon17,
    CelebA,
    ColoredMNIST,
    ISIC,
    ImageTform,
    MNISTColorizer,
    NICO,
    PACS,
    SSRP,
    Waterbirds,
)
from conduit.fair.data.datasets import ACSDataset, DummyDataset


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
def test_vision_datamodules(root: Path, dm: Type[CdtVisionDataModule]) -> None:
    dm_ = dm(root=root)
    dm_.prepare_data()
    dm_.setup()
    _ = next(iter(dm_.train_dataloader()))


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


def test_str_for_enum(root: Path) -> None:
    """Confirm that the conversion from ``str`` to ``Enum`` works."""
    with suppress(FileNotFoundError):
        Camelyon17(
            root,
            download=False,
            superclass="tumor",
            subclass="center",
            split_scheme="official",
            split="id_val",
        )

    with suppress(FileNotFoundError):
        CelebA(root, download=False, superclass="Smiling", subclass="Male", split="val")

    with suppress(FileNotFoundError):
        NICO(root, download=False, superclass="animals")

    with suppress(FileNotFoundError):
        PACS(root, download=False, domains="photo")

    with suppress(FileNotFoundError):
        Ecoacoustics(str(root), download=False, target_attrs=[SoundscapeAttr.N0])

    with suppress(FileNotFoundError):
        SSRP(str(root), download=False, split="Pre_Train")


@pytest.mark.slow
@pytest.mark.parametrize("segment_len", [1, 15, 30])
def test_audio_dataset(root: Path, segment_len: float) -> None:
    ds = Ecoacoustics(
        root=str(root),
        download=True,
        target_attrs=[SoundscapeAttr.HABITAT, SoundscapeAttr.SITE],
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
def test_ecoacoustics_metadata_labels(root: Path):
    target_attr = [SoundscapeAttr.HABITAT]
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
        target_attrs=[SoundscapeAttr.HABITAT],
        # train_tf=AT.Spectrogram(),
    )
    dm.prepare_data()
    dm.setup()

    # Test loading a sample.
    train_dl = dm.train_dataloader()
    test_sample = next(iter(train_dl))

    # Test size().
    assert test_sample.x.size()[1] == dm.dim_x[0]
    assert test_sample.x.size()[2] == dm.dim_x[1]
    assert test_sample.x.size()[3] == dm.dim_x[2]


@pytest.mark.slow
@pytest.mark.parametrize("train_batch_size", [1, 2, 8])
def test_ecoacoustics_dm_batch_multi_label(root: Path, train_batch_size: int) -> None:
    target_attrs = [SoundscapeAttr.HABITAT, SoundscapeAttr.SITE]
    data_module = EcoacousticsDataModule(
        root=str(root),
        segment_len=30.0,
        train_batch_size=train_batch_size,
        target_attrs=target_attrs,
        # train_tf=AT.Spectrogram(),
    )
    data_module.prepare_data()
    data_module.setup()

    train_dl = data_module.train_dataloader()
    sample = next(iter(train_dl))

    assert sample.y.size(1) == len(target_attrs)
    assert sample.x.size(0) == sample.y.size(0) == train_batch_size


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


def test_sample_add() -> None:
    x = torch.rand(3, 2)
    s = torch.randint(0, 2, (1,))
    y = torch.randint(0, 2, (1,))

    ns = NamedSample(x=x)
    assert (ns + ns).x.shape[0] == 6

    bs = BinarySample(x=x, y=y)
    assert (bs + bs).y.shape[0] == 2
    assert (bs + bs).x.shape[0] == 2

    ss = SubgroupSample(x=x, s=s)
    assert (ss + ss).s.shape[0] == 2
    assert (ss + ss).x.shape[0] == 2

    ts = TernarySample(x=x, s=s, y=y)
    assert (ts + ts).s.shape[0] == 2
    assert (ts + ts).x.shape[0] == 2


def test_sample_add_batched() -> None:
    x_batched = torch.rand(3, 2)
    s_batched = torch.randint(0, 2, (3, 1))
    y_batched = torch.randint(0, 2, (3, 1))

    ns_batched = NamedSample(x=x_batched)
    assert (ns_batched + ns_batched).x.shape[0] == 6

    bs_batched = BinarySample(x=x_batched, y=y_batched)
    assert (bs_batched + bs_batched).y.shape[0] == 6
    assert (bs_batched + bs_batched).x.shape[0] == 6

    ss_batched = SubgroupSample(x=x_batched, s=s_batched)
    assert (ss_batched + ss_batched).s.shape[0] == 6
    assert (ss_batched + ss_batched).x.shape[0] == 6

    ts_batched = TernarySample(x=x_batched, s=s_batched, y=y_batched)
    assert (ts_batched + ts_batched).s.shape[0] == 6
    assert (ts_batched + ts_batched).x.shape[0] == 6


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
            assert isinstance(sample, (SubgroupSample, TernarySample))
            assert sample.s.shape == (batch_size,)
        if y_card is None:
            assert isinstance(sample, (NamedSample, SubgroupSample))
        else:
            assert isinstance(sample, (BinarySample, TernarySample))
            assert sample.y.shape == (batch_size,)
        assert dm.train_data.feature_groups is not None
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
        eval_batch_size=batch_size,
        height=size,
        width=size,
        s_card=s_card,
        y_card=y_card,
        channels=channels,
        train_tf=transforms,
        test_tf=transforms,
    )
    dm.prepare_data()
    dm.setup()
    for loader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        sample = next(iter(loader))
        assert sample.x.shape == (batch_size, channels, size, size)
        if s_card is None:
            assert isinstance(sample, (NamedSample, BinarySample))
        else:
            assert isinstance(sample, (SubgroupSample, TernarySample))
            assert sample.s.shape == (batch_size,)
        if y_card is None:
            assert isinstance(sample, (NamedSample, SubgroupSample))
        else:
            assert isinstance(sample, (BinarySample, TernarySample))
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


def test_acs_dataset() -> None:
    acs_income = ACSDataset(setting=ACSDataset.Setting.income)
    assert acs_income.feature_groups is not None
    assert acs_income.feature_groups[0] == slice(2, 10)
    assert acs_income.x.shape == (22_268, 729)
    assert acs_income.s.shape == (22_268,)
    assert acs_income.y.shape == (22_268,)
    assert acs_income.cont_indexes == [0, 1]
