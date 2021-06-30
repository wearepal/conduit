from torch import Tensor, nn


class CrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _target = target.view(-1).long()
        return super().forward(input, _target)


class ClfL1(nn.L1Loss):
    def __init__(
        self, out_dim: int, size_average=None, reduce=None, reduction: str = 'mean'
    ) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.out_dim = out_dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _target = target.view(-1, self.out_dim).float()
        return super().forward(input, _target)
