import torch
from torch import Tensor
from torchvision.models import resnet

__all__ = ["concat_all_gather"]


@torch.no_grad()
def concat_all_gather(tensor: Tensor) -> Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]  # type: ignore
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)  # type: ignore

    return torch.cat(tensors_gather, dim=0)
