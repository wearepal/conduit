from typing import Dict, List, Union

import numpy as np
from torch import Tensor, nn

__all__ = [
    "cosine_scheduler",
    "get_params_groups",
]


def get_params_groups(model: nn.Module) -> List[Dict[str, Union[List[Tensor], float]]]:
    regularized: List[Tensor] = []
    not_regularized: List[Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]


def cosine_scheduler(
    *,
    base_value: float,
    final_value: float,
    total_iters: int,
    warmup_iters: int = 0,
    start_warmup_value: float = 0,
) -> np.ndarray:
    assert warmup_iters < total_iters
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule
