from typing import Optional, Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import Self, override

from conduit.types import Indexable, IndexType, Sized

__all__ = ["MemoryBank"]


@torch.no_grad()
def l2_hypersphere_init(capacity: int, *, dim: int) -> Tensor:
    return F.normalize(torch.randn(capacity, dim), dim=1)


class MemoryBank(nn.Module, Indexable, Sized):
    """Fixed-sized Memory Bank implementing a FIFO-update rule."""

    memory: Tensor

    def __init__(self, memory: Tensor) -> None:
        super().__init__()
        self.capacity = memory.size(0)
        self.dim = memory.shape[1:]
        self.register_buffer(name="memory", tensor=memory)
        self._ptr_pos = 0

    @classmethod
    @torch.no_grad()
    def with_l2_hypersphere_init(cls, capacity: int, *, dim: int) -> Self:
        return MemoryBank(l2_hypersphere_init(capacity=capacity, dim=dim))

    @classmethod
    @torch.no_grad()
    def with_randint_init(cls, capacity: int, *, dim: int, high: int, low: int = 0) -> Self:
        return MemoryBank(torch.randint(low=low, high=high, size=(capacity, dim)))

    @classmethod
    @torch.no_grad()
    def with_constant_init(
        cls: Type[Self],
        capacity: int,
        *,
        dim: int,
        value: float,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        return MemoryBank(torch.full(size=(capacity, dim), fill_value=value, dtype=dtype))

    def clone(self) -> Tensor:
        return self.memory.clone()

    @torch.no_grad()
    def push(self, values: Tensor) -> None:
        values = values.detach()
        num_values = values.size(0)
        # Coerce the values into being 2-dimensional.
        values = values.view(num_values, -1)
        residual = self.capacity - self._ptr_pos
        # Advance the pointer by 'num_values' steps.
        new_ptr_pos = (self._ptr_pos + num_values) % self.capacity

        if residual <= num_values:
            self.memory[self._ptr_pos :] = values[:residual]
            self.memory[:new_ptr_pos] = values[residual:]
        else:
            self.memory[self._ptr_pos : new_ptr_pos] = values
        self._ptr_pos = new_ptr_pos

    @override
    def __getitem__(self, index: IndexType) -> Tensor:
        return self.memory[index]

    @override
    def __len__(self) -> int:
        return len(self.memory)
