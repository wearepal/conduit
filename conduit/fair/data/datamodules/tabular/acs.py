from dataclasses import dataclass, field
from typing_extensions import override

from conduit.data import TrainValTestSplit
from conduit.data.datamodules.tabular import CdtTabularDataModule
from conduit.fair.data.datasets.acs import (
    ACSDataset,
    ACSHorizon,
    ACSSetting,
    ACSState,
    ACSSurvey,
    ACSSurveyYear,
)
from conduit.transforms import TabularNormalize, ZScoreNormalize

__all__ = ["ACSDataModule"]


@dataclass(kw_only=True)
class ACSDataModule(CdtTabularDataModule[ACSDataset]):
    setting: ACSSetting
    survey_year: ACSSurveyYear = ACSSurveyYear.YEAR_2018
    horizon: ACSHorizon = ACSHorizon.ONE_YEAR
    survey: ACSSurvey = ACSSurvey.PERSON
    states: list[ACSState] = field(default_factory=lambda: [ACSState.AL])
    class_train_props: dict[int, dict[int, float]] | None = None

    @override
    def _get_splits(self) -> TrainValTestSplit[ACSDataset]:
        data = ACSDataset(
            setting=self.setting,
            survey_year=self.survey_year,
            horizon=self.horizon,
            survey=self.survey,
            states=self.states,
        )
        train_data, val_data, test_data = data.subsampled_split(
            train_props=self.class_train_props,
            val_prop=self.val_prop,
            test_prop=self.test_prop,
            seed=self.seed,
        )
        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

    @override
    def _default_transforms(self) -> TabularNormalize:
        return ZScoreNormalize()
