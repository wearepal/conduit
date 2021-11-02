import pytest
import torch
from torch import Tensor
from torch.optim import AdamW

from conduit.optimizers import LAMB
from conduit.optimizers.sam import SAM


@pytest.mark.parametrize("adam", [True, False])
@pytest.mark.parametrize("debias", [True, False])
def test_lamb(adam: bool, debias: bool):
    params = torch.randn(10, requires_grad=True)
    optimizer = LAMB(params=[params], adam=adam, debias=debias)
    loss = params.norm()
    loss.backward()
    old_params = params.data.clone()
    optimizer.step()
    assert not torch.allclose(old_params.data, params.data)


@pytest.mark.parametrize("adaptive", [True, False])
def test_sam(adaptive: bool):
    params = torch.randn(10, requires_grad=True)
    base_optimizer = AdamW([params])
    optimizer = SAM(base_optimizer=base_optimizer, adaptive=adaptive)

    def _closure() -> Tensor:
        return params.norm()

    loss = _closure()
    loss.backward()
    old_params = params.data.clone()
    optimizer.step(closure=_closure)
    assert not torch.allclose(old_params.data, params.data)
