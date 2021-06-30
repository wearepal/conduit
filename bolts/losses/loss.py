from torch import Tensor, nn

__all__ = ["CrossEntropy"]


class CrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _target = target.view(-1).long()
        return super().forward(input, _target)
