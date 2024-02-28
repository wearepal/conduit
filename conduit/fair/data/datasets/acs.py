from collections.abc import Iterable
from enum import Enum, auto
from typing import TypeAlias
from typing_extensions import Self, TypeAliasType

from folktables import ACSDataSource, BasicProblem, generate_categories
import numpy as np
from numpy import typing as npt
import pandas as pd
from torch import Tensor

from conduit.data.datasets import random_split, stratified_split
from conduit.data.datasets.tabular.base import CdtTabularDataset
from conduit.data.structures import TernarySample

__all__ = ["ACSSetting", "ACSDataset", "ACSState", "ACSHorizon", "ACSSurvey", "ACSSurveyYear"]


class ACSSurveyYear(Enum):
    YEAR_2014 = "2014"
    YEAR_2015 = "2015"
    YEAR_2016 = "2016"
    YEAR_2017 = "2017"
    YEAR_2018 = "2018"


class ACSState(Enum):
    AL = auto()
    AK = auto()
    AZ = auto()
    AR = auto()
    CA = auto()
    CO = auto()
    CT = auto()
    DE = auto()
    FL = auto()
    GA = auto()
    HI = auto()
    ID = auto()
    IL = auto()
    IN = auto()
    IA = auto()
    KS = auto()
    KY = auto()
    LA = auto()
    ME = auto()
    MD = auto()
    MA = auto()
    MI = auto()
    MN = auto()
    MS = auto()
    MO = auto()
    MT = auto()
    NE = auto()
    NV = auto()
    NH = auto()
    NJ = auto()
    NM = auto()
    NY = auto()
    NC = auto()
    ND = auto()
    OH = auto()
    OK = auto()
    OR = auto()
    PA = auto()
    RI = auto()
    SC = auto()
    SD = auto()
    TN = auto()
    TX = auto()
    UT = auto()
    VT = auto()
    VA = auto()
    WA = auto()
    WV = auto()
    WI = auto()
    WY = auto()
    PR = auto()


class ACSHorizon(Enum):
    ONE_YEAR = "1-Year"
    FIVE_YEARS = "5-Year"


class ACSSurvey(Enum):
    PERSON = "person"
    HOUSEHOLD = "household"


def _employment_filter(data: pd.DataFrame) -> pd.DataFrame:
    """Custom filter.

    (Age) must be greater than 16 and less than 90, and (Person weight) must be
    greater than or equal to 1.
    """
    df = data
    return df[(df['AGEP'] > 16) & (df['AGEP'] < 90) & (df['PWGTP'] >= 1)]


def _adult_filter(data: pd.DataFrame) -> pd.DataFrame:
    """Mimic the filters in place for Adult data.

    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    return df[(df['AGEP'] > 16) & (df['PINCP'] > 100) & (df['WKHP'] > 0) & (df['PWGTP'] >= 1)]


_ACSEmploymentDisability = BasicProblem(
    features=[
        'AGEP',  # age; for range of values of features please check Appendix B.4 of Retiring Adult:
        # New Datasets for Fair Machine Learning NeurIPS 2021 paper
        'SCHL',  # educational attainment
        'MAR',  # marital status
        'RELP',  # relationship
        'ESP',  # employment status of parents
        'CIT',  # citizenship status
        'MIG',  # mobility status (lived here 1 year ago)
        'MIL',  # military service
        'ANC',  # ancestry recode
        'NATIVITY',  # nativity
        'DEAR',  # hearing difficulty
        'DEYE',  # vision difficulty
        'DREM',  # cognitive difficulty
        'SEX',  # sex
        'RAC1P',  # recoded detailed race code
        'GCL',  # grandparents living with grandchildren
    ],
    target='ESR',  # employment status recode
    target_transform=lambda x: x == 1,
    group='DIS',  # disability recode
    preprocess=_employment_filter,
    postprocess=lambda x: np.nan_to_num(x, nan=-1),
)
_ACSEmployment = BasicProblem(
    features=[
        'AGEP',  # age; for range of values of features please check Appendix B.4 of Retiring Adult:
        # New Datasets for Fair Machine Learning NeurIPS 2021 paper
        'SCHL',  # educational attainment
        'MAR',  # marital status
        'RELP',  # relationship
        'DIS',  # disability recode
        'ESP',  # employment status of parents
        'CIT',  # citizenship status
        'MIG',  # mobility status (lived here 1 year ago)
        'MIL',  # military service
        'ANC',  # ancestry recode
        'NATIVITY',  # nativity
        'DEAR',  # hearing difficulty
        'DEYE',  # vision difficulty
        'DREM',  # cognitive difficulty
        'SEX',  # sex
    ],
    target='ESR',
    target_transform=lambda x: x == 1,
    group='RAC1P',  # recoded detailed race code
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, nan=-1),
)
_ACSIncome = BasicProblem(
    features=[
        'AGEP',  # age; for range of values of features please check Appendix B.4 of Retiring Adult:
        # New Datasets for Fair Machine Learning NeurIPS 2021 paper
        'COW',
        'SCHL',  # educational attainment
        'MAR',  # marital status
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',  # sex
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='RAC1P',  # recoded detailed race code
    preprocess=_adult_filter,
    postprocess=lambda x: np.nan_to_num(x, nan=-1),
)


class ACSSetting(Enum):
    employment = _ACSEmployment
    income = _ACSIncome
    employment_disability = _ACSEmploymentDisability


class ACSDataset(CdtTabularDataset[TernarySample, Tensor, Tensor]):
    """Wrapper for the ACS dataset from Folktables."""

    Setting: TypeAlias = ACSSetting
    Horizon: TypeAlias = ACSHorizon
    State: TypeAlias = ACSState
    Survey: TypeAlias = ACSSurvey
    SurveyYear: TypeAlias = ACSSurveyYear

    def __init__(
        self,
        setting: ACSSetting,
        survey_year: SurveyYear = SurveyYear.YEAR_2018,
        horizon: Horizon = Horizon.ONE_YEAR,
        survey: ACSSurvey = ACSSurvey.PERSON,
        states: Iterable[ACSState] = (ACSState.AL,),
    ):
        data_source = ACSDataSource(
            survey_year=survey_year.value, horizon=horizon.value, survey=survey.value
        )
        acs_data = data_source.get_data(states=[state.name for state in states], download=True)
        dataset: BasicProblem = setting.value

        # `generate_categories` is only available for years >= 2017.
        group: npt.NDArray
        if int(survey_year.value) >= 2017:
            categories = generate_categories(
                features=dataset.features,
                definition_df=data_source.get_definitions(download=True),
            )

            # One-hot encoding based on the categories.
            features_df, label_df, group_df = dataset.df_to_pandas(
                acs_data, categories=categories, dummies=True
            )

            feature_groups, ohe_indexes = feature_groups_from_categories(categories, features_df)
            label = label_df.to_numpy(dtype=np.int64)
            group = group_df.to_numpy(dtype=np.int64)
            features = features_df.to_numpy(dtype=np.float32)
            non_ohe_indexes = list(set(range(features.shape[1])) - set(ohe_indexes))
        else:
            # Categorical features are *not* one-hot encoded for years < 2017.
            features, label, group = dataset.df_to_numpy(acs_data)
            non_ohe_indexes, feature_groups = None, None

        # Make sure `group` is zero-indexed.
        group -= group.min()
        s_values = np.unique(group)
        assert np.array_equal(s_values, np.arange(len(s_values))), "Group vals should be contiguous"

        super().__init__(
            x=features,
            y=label,
            s=group,
            non_ohe_indexes=non_ohe_indexes,
            feature_groups=feature_groups,
        )

    def subsampled_split(
        self,
        train_props: dict[int, dict[int, float]] | None,
        *,
        val_prop: float,
        test_prop: float,
        seed: int,
    ) -> tuple[Self, Self, Self]:
        """Create a split of the dataset in which the training set is missing certain groups.

        :param train_props: Specification for which groups to drop from the training set.
        :param val_prop: Proportion of the data to use for the validation set.
        :param test_prop: Proportion of the data to use for the test set.
        :param seed: PRNG seed to use for splitting the data.
        :returns: Random subsets of the data of the requested proportions.
        """
        val_test_prop = val_prop + test_prop
        train_data, val_test_data = stratified_split(
            self,
            default_train_prop=1 - val_test_prop,
            train_props=train_props,
            seed=seed,
            reproducible=True,
        )
        val_data, test_data = random_split(
            val_test_data, props=val_prop / val_test_prop, seed=seed, reproducible=True
        )
        return train_data, val_data, test_data


CategoryName = TypeAliasType("CategoryName", str)
ValueName = TypeAliasType("ValueName", str)


def feature_groups_from_categories(
    categories: dict[CategoryName, dict[float, ValueName]], features: pd.DataFrame
) -> tuple[list[slice], list[int]]:
    slices: list[slice] = []
    disc_indexes: list[int] = []

    for category_name, value_entries in categories.items():
        indexes = []
        for value_name in value_entries.values():
            feature_name = f"{category_name}_{value_name}"
            if feature_name in features.columns:
                feature_index = features.columns.get_loc(feature_name)
                indexes.append(feature_index)

        # Determine the slice bounds for this category.
        start = min(indexes)
        stop = max(indexes) + 1

        # Check that the indexes are contiguous.
        index_set = set(indexes)
        for i in range(start, stop):
            assert i in index_set

        slices.append(slice(start, stop))
        disc_indexes.extend(indexes)
    return slices, disc_indexes
