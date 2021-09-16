import logging
from typing import ClassVar, List, Optional, Sequence, Tuple, TypeVar, Union, cast

from kit import implements
from kit.torch.data import prop_random_split
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import final

from conduit.data.structures import (
    BinarySample,
    InputData,
    NamedSample,
    SubgroupSample,
    TargetData,
    TernarySample,
)

__all__ = ["CdtDataset", "D"]


D = TypeVar("D", bound="CdtDataset")


class CdtDataset(Dataset):
    _repr_indent: ClassVar[int] = 4
    _logger: Optional[logging.Logger] = None

    def __init__(
        self, *, x: InputData, y: Optional[TargetData] = None, s: Optional[TargetData] = None
    ) -> None:
        self.x = x
        if isinstance(y, np.ndarray):
            y = torch.as_tensor(y)
        if isinstance(s, np.ndarray):
            s = torch.as_tensor(s)
        self.y = y if y is None else y.squeeze()
        self.s = s if s is None else s.squeeze()

        self._dim_x: Optional[torch.Size] = None
        self._dim_s: Optional[torch.Size] = None
        self._dim_y: Optional[torch.Size] = None
        self._card_y: Optional[int] = None
        self._card_s: Optional[int] = None

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    @staticmethod
    def extra_repr() -> str:
        return ""

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    def _sample_x(self, index: int, *, coerce_to_tensor: bool = False) -> Tensor:
        x = self.x[index]
        if coerce_to_tensor and (not isinstance(x, Tensor)):
            x = torch.as_tensor(x)
        return x

    def _sample_s(self, index: int) -> Optional[Tensor]:
        if self.s is None:
            return None
        return self.s[index]

    def _sample_y(self, index: int) -> Optional[Tensor]:
        if self.y is None:
            return None
        return self.y[index]

    @property
    @final
    def dim_x(
        self,
    ) -> Tuple[int, ...]:
        if self._dim_x is None:
            self._dim_x = self._sample_x(0, coerce_to_tensor=True).shape
        return self._dim_x

    @property
    @final
    def dim_s(
        self,
    ) -> Tuple[int, ...]:
        if self.s is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.dim_s' cannot be determined as '{cls_name}.s' is 'None'"
            )
        elif self._dim_s is None:
            self._dim_s = torch.Size((1,)) if self.s.ndim == 1 else self.s.shape[1:]
        return self._dim_s

    @property
    @final
    def dim_y(
        self,
    ) -> Tuple[int, ...]:
        if self.y is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.dim_y' cannot be determined as '{cls_name}.y' is 'None'"
            )
        elif self._dim_y is None:
            self._dim_y = torch.Size((1,)) if self.y.ndim == 1 else self.y.shape[1:]
        return self._dim_y

    @property
    @final
    def card_y(
        self,
    ) -> int:
        if self.y is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.card_y' cannot be determined as '{cls_name}.y' is 'None'"
            )
        elif self._card_y is None:
            self._card_y = len(self.y.unique())
        return self._card_y

    @property
    @final
    def card_s(
        self,
    ) -> int:
        if self.s is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.card_s' cannot be determined as '{cls_name}.s' is 'None'"
            )
        elif self._card_s is None:
            self._card_s = len(self.s.unique())
        return self._card_s

    @implements(Dataset)
    @final
    def __getitem__(
        self, index: int
    ) -> Union[NamedSample, BinarySample, SubgroupSample, TernarySample]:
        x = self._sample_x(index)
        y = self._sample_y(index)
        s = self._sample_s(index)
        # Fetch the appropriate 'Sample' class
        if y is None:
            if s is None:
                return NamedSample(x=x)
            return SubgroupSample(x=x, s=s)
        if s is None:
            return BinarySample(x=x, y=y)
        return TernarySample(x=x, y=y, s=s)

    def __len__(self) -> int:
        return len(self.x)

    def make_subset(
        self: D,
        indices: Union[List[int], npt.NDArray[np.uint64], Tensor, slice],
        deep: bool = False,
    ) -> D:
        """Create a subset of the dataset from the given indices.

        :param indices: The sample-indices from which to create the subset.
        In the case of being a numpy array or tensor, said array or tensor
        must be 0- or 1-dimensional.

        :param deep: Whether to create a copy of the underlying dataset as
        a basis for the subset. If False then the data of the subset will be
        a view of original dataset's data.

        :returns: A subset of the dataset from the given indices.
        """
        # lazily import make_subset to prevent it being a circular import
        from conduit.data.datasets.utils import make_subset

        return make_subset(dataset=self, indices=indices, deep=deep)

    def random_split(self: D, props: Union[Sequence[float], float], deep: bool = False) -> List[D]:
        """Randomly split the dataset into subsets according to the given proportions.

        :param props: The fractional size of each subset into which to randomly split the data.
        Elements must be non-negative and sum to 1 or less; if less then the size of the final
        split will be computed by complement.

        :param deep: Whether to create a copy of the underlying dataset as
        a basis for the random subsets. If False then the data of the subsets will be
        views of original dataset's data.

        :returns: Random subsets of the data of the requested proportions.
        """
        # lazily import make_subset to prevent it being a circular import
        from conduit.data.datasets.utils import make_subset

        splits = prop_random_split(dataset=self, props=props)
        splits = cast(List[D], [make_subset(split, indices=None, deep=deep) for split in splits])
        return splits
