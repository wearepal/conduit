import inspect
from typing import Any, Callable, Dict, List, TypeVar

from pytorch_lightning.utilities.types import STEP_OUTPUT
from ranzen.misc import gcopy
import torch
from torch import Tensor, nn

__all__ = [
    "PartialModule",
    "aggregate_over_epoch",
    "decorate_all_methods",
    "make_no_grad",
    "prefix_keys",
]
EPOCH_OUTPUT = List[STEP_OUTPUT]


@torch.no_grad()
def aggregate_over_epoch(outputs: EPOCH_OUTPUT, *, metric: str) -> Tensor:
    return torch.cat([step_output[metric] for step_output in outputs])  # type: ignore


def prefix_keys(dict_: Dict[str, Any], *, prefix: str, sep: str = "/") -> Dict[str, Any]:
    """Prepend the prefix to all keys of the dict"""
    return {f"{prefix}{sep}{key}": value for key, value in dict_.items()}


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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs, **self.bound_kwargs)
