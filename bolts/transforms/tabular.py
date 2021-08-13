from __future__ import annotations
from abc import abstractmethod
import math
from typing import ClassVar

from kit import implements
import torch
from torch import Tensor

__all__ = [
    "MinMaxNormalization",
    "QuantileNormalization",
    "TabularTransform",
    "ZScoreNormalization",
]


class TabularTransform:
    _EPS: ClassVar[float] = torch.finfo(torch.float32).eps

    def __init__(self, inplace: bool = False, indices: slice | list[int] = slice(None)) -> None:
        self.inplace = inplace
        self.col_indexes = indices
        self._is_fitted = False

    def is_fitted(self):
        return self._is_fitted

    def _maybe_clone(self, data: Tensor) -> Tensor:
        if not self.inplace:
            data = data.clone()
        return data

    @abstractmethod
    def _fit(self, data: Tensor) -> None:
        """inplace operation."""
        ...

    def fit(self, data: Tensor) -> TabularTransform:
        self._fit(data[:, self.col_indexes])
        self._is_fitted = False
        return self

    def fit_transform(self, data: Tensor) -> Tensor:
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def _inverse_transform(self, data: Tensor) -> None:
        """inplace operation."""
        ...

    def inverse_transform(self, data: Tensor) -> Tensor:
        data = self._maybe_clone(data)
        self._inverse_transform(data[:, self.col_indexes])
        return data

    @abstractmethod
    def _transform(self, data: Tensor) -> None:
        """inplace operation."""
        ...

    def transform(self, data: Tensor) -> Tensor:
        data = self._maybe_clone(data)
        self._transform(data[:, self.col_indexes])
        return data

    def __call__(self, data: Tensor) -> Tensor:
        return self.transform(data)


class ZScoreNormalization(TabularTransform):

    mean: Tensor
    std: Tensor

    @implements(TabularTransform)
    def _fit(self, data: Tensor) -> None:
        self.std, self.mean = torch.std_mean(data, dim=0, keepdim=True, unbiased=True)

    @implements(TabularTransform)
    def _inverse_transform(self, data: Tensor) -> None:
        data *= self.std
        data += self.mean

    @implements(TabularTransform)
    def _transform(self, data: Tensor) -> None:
        data -= self.mean
        data /= self.std.clamp_min(self._EPS)


class QuantileNormalization(TabularTransform):

    iqr: Tensor
    median: Tensor

    def __init__(self, q_min: float = 0.25, q_max: float = 0.75, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)
        if not (0 <= q_min <= 1):
            raise ValueError("q_min must be in the range [0, 1]")
        if not (0 <= q_max <= 1):
            raise ValueError("q_max must be in the range [0, 1]")
        if q_min > q_max:
            raise ValueError("'q_min' cannot be greater than 'q_max'.")
        self.q_min = q_min
        self.q_max = q_max

    @staticmethod
    def _compute_quantile(q: float, sorted_values: Tensor) -> Tensor:
        q_ind_frac, q_ind_int = math.modf(q * (len(sorted_values) - 1))
        q_ind_int = int(q_ind_int)
        q_quantile = sorted_values[q_ind_int]
        if q_ind_frac > 0:
            q_quantile += (sorted_values[q_ind_int + 1] - q_quantile) * q_ind_frac
        return q_quantile

    @implements(TabularTransform)
    def _fit(self, data: Tensor) -> None:
        sorted_values = data.sort(dim=0, descending=False).values
        # Compute the 'lower quantile'
        q_min_quantile = self._compute_quantile(q=self.q_min, sorted_values=sorted_values)
        # Compute the 'upper quantile'
        q_max_quantile = self._compute_quantile(q=self.q_max, sorted_values=sorted_values)
        # Compute the interquantile range
        self.iqr = q_max_quantile - q_min_quantile
        self.median = self._compute_quantile(q=0.5, sorted_values=sorted_values)

    @implements(TabularTransform)
    def _inverse_transform(self, data: Tensor) -> None:
        data *= self.iqr
        data += self.median

    @implements(TabularTransform)
    def _transform(self, data: Tensor) -> None:
        data -= self.median
        data /= self.iqr.clamp_min(self._EPS)


class MinMaxNormalization(TabularTransform):

    orig_max: Tensor
    orig_min: Tensor
    orig_range: Tensor

    def __init__(self, new_min: float = 0.0, new_max: float = 1.0, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)
        if new_min > new_max:
            raise ValueError("'new_min' cannot be greater than 'new_max'.")
        self.new_min = new_min
        self.new_max = new_max
        self.new_range = self.new_max - self.new_min

    @implements(TabularTransform)
    def _fit(self, data: Tensor) -> None:
        self.orig_min = torch.min(data, dim=0, keepdim=True).values
        self.orig_max = torch.max(data, dim=0, keepdim=True).values
        self.orig_range = self.orig_max - self.orig_min

    @implements(TabularTransform)
    def _inverse_transform(self, data: Tensor) -> None:
        data -= self.new_min
        data /= self.new_range + self._EPS
        data *= self.orig_range
        data += self.orig_min

    @implements(TabularTransform)
    def _transform(self, data: Tensor) -> None:
        data -= self.orig_min
        data /= self.orig_range.clamp_min(self._EPS)
        data *= self.new_range
        data += self.new_min
