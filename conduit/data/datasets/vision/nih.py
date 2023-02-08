from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional, Union, cast

import pandas as pd
from ranzen import parsable, str_to_enum
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from conduit.data.structures import TernarySample

from .base import CdtVisionDataset
from .utils import ImageTform

__all__ = ["NIHChestXRays", "NIHSplit", "NIHSubgroup", "NIHTarget"]


class NIHSplit(Enum):
    TRAIN = "train_val_list.txt"
    TEST = "test_list.txt"


class NIHSubgroup(Enum):
    GENDER = "Patient Gender"
    AGE = "Patient Age"


class NIHTarget(Enum):
    """
    Taget attributes for the NIH chest X-rays dataset.

    Fraction of labels that are positive for each of the thoracic diseases:
        Atelectasis           0.103095
        Cardiomegaly          0.024759
        Consolidation         0.041625
        Edema                 0.020540
        Effusion              0.118775
        Emphysema             0.022440
        Fibrosis              0.015037
        Hernia                0.002025
        Infiltration          0.177435
        Mass                  0.051570
        Nodule                0.056466
        Pleural_Thickening    0.030191
        Pneumonia             0.012763
        Pneumothorax          0.04728
    """

    ATELECTASIS = "Atelectasis"
    CARDIOMEGALY = "Cardiomegaly"
    CONSOLIDATION = "Consolidation"
    EDEMA = "Edema"
    EFFUSION = "Effusion"
    EMPHYSEMA = "Emphysema"
    FIBROSIS = "Fibrosis"
    HERNIA = "Hernia"
    INFILTRATION = "Infiltration"
    MASS = "Mass"
    NODULE = "Nodule"
    PLEURAL_THICKENING = "Pleural_Thickening"
    PNEUMONIA = "Pneumonia"
    PNEUMOTHORAX = "Pneumothorax"
    FINDING = "No Finding"


class NIHChestXRays(CdtVisionDataset[TernarySample, Tensor, Tensor]):
    """ "
    National Institutes of Health (NIH) chest X-Rays dataset.
    This NIH Chest X-rays dataset is comprised of 112,120 X-ray images with disease labels from
    30,805 unique patients. To create these labels, the authors used Natural Language Processing to
    text-mine disease classifications from the associated radiological reports. The labels are
    expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology
    reports are not publicly available but you can find more details on the labeling process in
    `this
    <https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community>`__
    Open Access paper.

    The dataset can be downloaded by following the above link or from `kaggle <https://www.kaggle.com/datasets/nih-chest-xrays/data>`__
    """

    SampleType: TypeAlias = TernarySample
    Target: TypeAlias = NIHTarget
    Subgroup: TypeAlias = NIHSubgroup
    Split: TypeAlias = NIHSplit

    _METADATA_FILENAME: ClassVar[str] = "Data_Entry_2017.csv"
    _BASE_DIR_NAME: ClassVar[str] = "nih_chest_x_rays"

    @parsable
    def __init__(
        self,
        root: Union[Path, str],
        *,
        target: Optional[Union[NIHTarget, str]] = NIHTarget.FINDING,
        subgroup: Union[NIHSubgroup, str] = NIHSubgroup.GENDER,
        split: Optional[Union[NIHSplit, str]] = None,
        transform: Optional[ImageTform] = None,
        num_quantiles: Optional[float] = 4,
        download: bool = True,
    ) -> None:
        """
        :param root: Root directory of the dataset.
        :param target: Attribute to set as the target attribute ('y').
        :param subgroup: Attribute to set as the subgroup attribute ('s').
        :param split: Which predefined split of the dataset to use. If ``None`` then the full
            (unsplit) dataset will be returned.
        :param transform: A function/transform that takes in a PIL or ndarray image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        :param num_quantiles: Number of quantiles (equal-sized buckets) to bin 'Patient Age' into
            (only applicable when ``sens_attr=NIHSensAttr.AGE``). E.g. 10 for deciles, 4 for
            quartiles.
        :param download: If ``True``, downloads the dataset from the internet and puts it in the
            root directory. If the dataset is already downloaded, it is not downloaded again.
            This requires kaggle credentials.
        :raises FileNotFoundError: If ``download=False`` and an existing dataset cannot be found in
            the root directory.
        """
        self.root = Path(root)
        self._base_dir = self.root / self._BASE_DIR_NAME
        self._metadata_fp = self._base_dir / self._METADATA_FILENAME
        if target is not None:
            target = str_to_enum(str_=target, enum=NIHTarget)
        self.target = target
        self.subgroup = str_to_enum(str_=subgroup, enum=NIHSubgroup)
        if split is not None:
            split = str_to_enum(str_=split, enum=NIHSplit)
        self.split = split
        self.download = download
        if not self._check_unzipped():
            if self.download:
                import kaggle  # type: ignore

                kaggle.api.authenticate()
                kaggle.api.dataset_download_files(
                    dataset="nih-chest-xrays/data",
                    path=self._base_dir,
                    unzip=True,
                    quiet=False,
                )
            else:
                raise FileNotFoundError(
                    f"Data not found at location {self.root.resolve()}. Have you downloaded it?"
                )

        self.metadata = cast(pd.DataFrame, pd.read_csv(self._metadata_fp))
        if self.split is not None:
            split_info = pd.read_csv(
                self._base_dir / self.split.value, header=None, names=["Image Index"]
            )
            self.metadata = self.metadata.merge(split_info, on="Image Index")

        # In the case of Patient Gender, factorize yields the mapping: M -> 0, F -> 1
        s_pd = self.metadata[self.subgroup.value]
        if (self.subgroup is NIHSubgroup.AGE) and (num_quantiles is not None):
            s_pd = cast(pd.Series, pd.qcut(s_pd, q=num_quantiles))
        s_pd_le = s_pd.factorize()[0]
        s = torch.as_tensor(s_pd_le, dtype=torch.long)

        findings_str = self.metadata["Finding Labels"].str.split("|")
        self.encoder = MultiLabelBinarizer().fit(findings_str)
        findings_ml = pd.DataFrame(
            self.encoder.transform(findings_str), columns=self.encoder.classes_  # type: ignore
        )
        self.metadata = pd.concat((self.metadata, findings_ml), axis=1)
        if self.target is None:
            findings_ml.drop("No Finding", axis=1, inplace=True)
        else:
            findings_ml = findings_ml[self.target.value]
            if self.target is NIHTarget.FINDING:
                # Flip the label such that the presence of some thoracic disease
                # is represented by the positive class.
                findings_ml = 1 - findings_ml
        y = torch.as_tensor(findings_ml.to_numpy(), dtype=torch.long)

        filepaths = (
            pd.Series(self._base_dir.glob("*/*/*"), dtype=str)
            .sort_values()
            .str.split("/", expand=True, n=4)[4]
            .rename("Image Path")
        )
        if self.split is None:
            self.metadata = pd.concat((filepaths, self.metadata), axis=1)
        else:
            filenames = filepaths.str.rsplit("/", expand=True, n=1)[1].rename("Image Index")
            filepaths = pd.concat((filepaths, filenames), axis=1)
            self.metadata = self.metadata.merge(filepaths, on="Image Index")
        x = self.metadata["Image Path"].to_numpy()
        super().__init__(image_dir=self._base_dir, x=x, s=s, y=y, transform=transform)

    def _check_unzipped(self) -> bool:
        return (self._metadata_fp).exists()
