from enum import Enum
from functools import partial

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import resnet

__all__ = ["concat_all_gather", "MemoryBank", "ResNetArch"]


@torch.no_grad()
def concat_all_gather(tensor: Tensor) -> Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]  # type: ignore
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)  # type: ignore

    return torch.cat(tensors_gather, dim=0)


class MemoryBank(nn.Module):
    def __init__(self, *, dim: int, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity
        self._ptr_pos = 0
        self.memory: Tensor
        self.register_buffer(name="memory", tensor=self._init_memory(dim=dim))

    @torch.no_grad()
    def _init_memory(self, dim: int) -> Tensor:
        return F.normalize(torch.randn(self.capacity, dim), dim=0)

    @torch.no_grad()
    def push(self, values: Tensor) -> None:
        values = values.detach()
        num_values = values.size(0)
        residual = (self.capacity - 1) - self._ptr_pos
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


class ResNetArch(Enum):
    resnet18 = partial(resnet.resnet18)
    resnet50 = partial(resnet.resnet50)
    resnet34 = partial(resnet.resnet34)
