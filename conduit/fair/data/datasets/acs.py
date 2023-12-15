from collections.abc import Iterable
from enum import Enum, auto
from typing import TypeAlias

from folktables import ACSDataSource, ACSEmployment, ACSIncome, BasicProblem, generate_categories
import numpy as np
import pandas as pd
from torch import Tensor

from conduit.data.datasets.tabular.base import CdtTabularDataset
from conduit.data.structures import TernarySample
from conduit.transforms.tabular import TabularTransform

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


class ACSSetting(Enum):
    employment = ACSEmployment
    income = ACSIncome


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
        transform: TabularTransform | None = None,
        target_transform: TabularTransform | None = None,
    ):
        data_source = ACSDataSource(
            survey_year=survey_year.value, horizon=horizon.value, survey=survey.value
        )
        acs_data = data_source.get_data(states=[state.name for state in states], download=True)
        dataset: BasicProblem = setting.value

        # `generate_categories` is only available for years >= 2017.
        if int(survey_year.value) >= 2017:
            categories = generate_categories(
                features=dataset.features,
                definition_df=data_source.get_definitions(download=True),
            )

            # One-hot encoding based on the categories.
            features_df, label_df, group_df = dataset.df_to_pandas(
                acs_data, categories=categories, dummies=True
            )

            feature_groups, disc_indexes = feature_groups_from_categories(categories, features_df)
            label = label_df.to_numpy(dtype=np.int64)
            group = group_df.to_numpy(dtype=np.int64)
            features = features_df.to_numpy(dtype=np.float32)
            cont_indexes = list(set(range(features.shape[1])) - set(disc_indexes))
        else:
            # Categorical features are *not* one-hot encoded for years < 2017.
            features, label, group = dataset.df_to_numpy(acs_data)
            cont_indexes, disc_indexes, feature_groups = None, None, None

        super().__init__(
            x=features,
            y=label,
            s=group,
            transform=transform,
            target_transform=target_transform,
            cont_indexes=cont_indexes,
            disc_indexes=disc_indexes,
            feature_groups=feature_groups,
        )


CategoryName: TypeAlias = str
ValueName: TypeAlias = str


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
