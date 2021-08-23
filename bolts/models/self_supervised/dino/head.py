from __future__ import annotations
from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .utils import trunc_normal_
from .vit import VisionTransformer

__all__ = [
    "DINOHead",
    "MultiCropNet",
    "MultiCropWrapper",
]


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)  # type: ignore
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone: VisionTransformer, head: DINOHead) -> None:
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x: list[Tensor]) -> Tensor:
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            output: Tensor = _out if start_idx == 0 else torch.cat((output, _out))  # type: ignore
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)  # type: ignore


class MultiCropNet(nn.Module):
    def __init__(
        self,
        arch_fn: Callable[[int], VisionTransformer],
        patch_size: int,
        norm_last_layer: bool,
        use_bn_in_head: bool,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = arch_fn(patch_size)
        embed_dim = self.backbone.embed_dim
        self.head = DINOHead(
            embed_dim,
            out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )
        self.net = MultiCropWrapper(backbone=self.backbone, head=self.head)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
