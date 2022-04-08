import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["MemoryBank"]


class MemoryBank(nn.Module):
    """PyTorch-based FIFO queue for storing momentum-encoder encodings."""

    def __init__(self, *, dim: int, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity
        self._ptr_pos = 0
        self.memory: Tensor
        self.register_buffer(name="memory", tensor=self._init_memory(dim=dim))

    @torch.no_grad()
    def _init_memory(self, dim: int) -> Tensor:
        return F.normalize(torch.randn(self.capacity, dim), dim=1)

    @torch.no_grad()
    def push(self, values: Tensor) -> None:
        values = values.detach()
        num_values = values.size(0)
        residual = self.capacity - self._ptr_pos
        new_ptr_pos = (self._ptr_pos + num_values) % self.capacity  # move pointer

        if residual < num_values:
            self.memory[self._ptr_pos :] = values[:residual]
            self.memory[:new_ptr_pos] = values[residual:]
        else:
            self.memory[self._ptr_pos : new_ptr_pos] = values
        self._ptr_pos = new_ptr_pos

    def __getitem__(self, index: int) -> Tensor:
        return self.memory[index]

    def __len__(self) -> int:
        return len(self.memory)
