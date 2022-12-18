# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecoupledContrastiveLossConf:
    _target_: str = "conduit.models.self_supervised.loss.DecoupledContrastiveLoss"
    temperature: float = 0.1
    weight_fn: Any = None  # Optional[Callable[[Tensor, Tensor], Tensor]]
