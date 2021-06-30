from torch import Tensor, nn


class CrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _target = target.view(-1).long()
        return super().forward(input, _target)


class ClfL1(nn.L1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _target = target.view_as(input).float()
        return super().forward(input, _target)
