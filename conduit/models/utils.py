import inspect
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

from ranzen.misc import gcopy
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torch import Tensor, nn

__all__ = [
    "PartialModule",
    "accuracy",
    "aggregate_over_epoch",
    "decorate_all_methods",
    "make_no_grad",
    "precision_at_k",
    "prediction",
    "prefix_keys",
]


@torch.no_grad()
def aggregate_over_epoch(outputs: EPOCH_OUTPUT, *, metric: str) -> Tensor:
    return torch.cat([step_output[metric] for step_output in outputs])  # type: ignore


def prefix_keys(dict_: Dict[str, Any], *, prefix: str, sep: str = "/") -> Dict[str, Any]:
    """Prepend the prefix to all keys of the dict"""
    return {f"{prefix}{sep}{key}": value for key, value in dict_.items()}


@torch.no_grad()
def prediction(logits: Tensor) -> Tensor:
    logits = torch.atleast_1d(logits.squeeze())
    if logits.ndim == 1:
        return (logits > 0).long()
    return logits.argmax(dim=1)


@torch.no_grad()
def accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    logits = torch.atleast_1d(logits.squeeze())
    targets = torch.atleast_1d(targets.squeeze()).long()
    if len(logits) != len(targets):
        raise ValueError("'logits' and 'targets' must match in size at dimension 0.")
    preds = (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=1)
    return (preds == targets).float().mean()


@torch.no_grad()
def precision_at_k(
    logits: Tensor, targets: Tensor, top_k: Union[int, Tuple[int, ...]] = (1,)
) -> List[Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if isinstance(top_k, int):
        top_k = (top_k,)
    maxk = max(top_k)
    batch_size = targets.size(0)
    _, pred = logits.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res: List[Tensor] = []
    for k in top_k:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


T = TypeVar("T")


def decorate_all_methods(decorator: Callable[[Callable], Callable], obj: T) -> T:
    methods = inspect.getmembers(obj, inspect.ismethod)
    for name, method in methods:
        setattr(obj, name, decorator(method))
    return obj


U = TypeVar("U", bound=nn.Module)


def make_no_grad(module: U) -> U:
    return decorate_all_methods(torch.no_grad(), gcopy(module, deep=False))


class PartialModule(nn.Module):
    def __init__(self, func: Callable, **kwargs: Any):
        super().__init__()
        self.func = func
        self.bound_kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any):
        return self.func(*args, **kwargs, **self.bound_kwargs)
