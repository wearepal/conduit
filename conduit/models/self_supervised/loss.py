from typing import Callable, Optional, Type, TypeVar, cast

import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import Self

__all__ = [
    "DecoupledContrastiveLoss",
    "decoupled_constrastive_loss",
    "moco_loss",
    "suprvised_contrastive_loss",
]


def moco_loss(
    anchors: Tensor,
    *,
    positives: Tensor,
    negatives: Tensor,
    temperature: float = 1.0,
    dcl: bool = True,
) -> Tensor:
    # positive logits: (N,)
    l_pos = (anchors * positives.unsqueeze(1)).sum(-1).view(-1, 1) / temperature
    # negative logits: (N, K)
    l_neg = (anchors @ negatives.T).view(l_pos.size(0), -1) / temperature
    # Compute the partition function either according to the original InfoNCE formulation
    # or according to the DCL formulation which excludes the positive samples.
    z = l_neg.logsumexp(dim=1) if dcl else torch.cat([l_pos, l_neg], dim=1).logsumexp(dim=1)
    return (z - l_pos).mean()


T = TypeVar("T", Tensor, None)


def suprvised_contrastive_loss(
    anchors: Tensor,
    *,
    anchor_labels: Tensor,
    candidates: T = None,
    candidate_labels: T = None,
    temperature: float = 0.1,
    exclude_diagonal: bool = False,
) -> Tensor:
    # Create new variables for the candidate- variables to placate
    # the static-type checker.
    if candidates is None:
        candidates_t = anchors
        candidate_labels_t = anchor_labels
        # Forbid interactions between the samples and themsleves.
        exclude_diagonal = True
    else:
        candidates_t = candidates
        candidate_labels_t = cast(Tensor, candidate_labels)

    anchor_labels = anchor_labels.view(-1, 1)
    candidate_labels_t = candidate_labels_t.flatten()
    # The positive samples for a given anchor are those samples from the candidate set sharing its
    # label.
    mask = anchor_labels == candidate_labels_t
    if exclude_diagonal:
        mask.fill_diagonal_(False)
    pos_inds = mask.nonzero(as_tuple=True)
    row_inds, col_inds = pos_inds
    # Return early if there are no positive pairs.
    if len(row_inds) == 0:
        return anchors.new_zeros(())
    # Only compute the pairwise similarity for those samples which have positive pairs.
    selected_rows, row_inverse, row_counts = row_inds.unique(
        return_inverse=True, return_counts=True
    )
    logits = anchors[selected_rows] @ candidates_t.T
    # Apply temperature-scaling to the logits.
    logits /= temperature
    z = logits.logsumexp(dim=-1).flatten()
    # Tile the row counts if dealing with multicropping.
    if anchors.ndim == 3:
        row_counts = row_counts.unsqueeze(1).expand(-1, anchors.size(1))
    counts_flat = row_counts[row_inverse].flatten()
    positives = logits[row_inverse, ..., col_inds].flatten() / counts_flat
    return (z.sum() - positives.sum()) / z.numel()


def decoupled_constrastive_loss(
    z1: Tensor,
    z2: Tensor,
    *,
    temperature: float = 0.1,
    weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> Tensor:
    """
    Calculates the one-way `decoupled constrastive loss <https://arxiv.org/pdf/2110.06848.pdf>`_.

    :param z1: First embedding vector
    :param z2: Second embedding vector
    :param weight_fn: The weighting function of the positive sample loss.
    :param temperature: Temperature controlling the sharpness of the softmax distribution.

    :return: One-way loss between the embedding vectors.
    """
    cross_view_distance = torch.mm(z1, z2.t())
    positive_loss = -torch.diag(cross_view_distance) / temperature
    if weight_fn is not None:
        positive_loss = positive_loss * weight_fn(z1, z2)
    neg_similarity = torch.cat((z1 @ z1.t(), cross_view_distance), dim=1) / temperature
    neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
    eps = torch.finfo(z1.dtype).eps
    negative_loss = torch.logsumexp(neg_similarity + neg_mask * eps, dim=1, keepdim=False)
    return (positive_loss + negative_loss).mean()


def _von_mises_fisher_weighting(z1: Tensor, z2: Tensor, *, sigma: float = 0.5) -> Tensor:
    return 2 - len(z1) * ((z1 * z2).sum(dim=1) / sigma).softmax(dim=0).squeeze()


class DecoupledContrastiveLoss(nn.Module):
    """
    Implementation of the Decoupled Contrastive Loss proposed in
    https://arxiv.org/pdf/2110.06848.pdf, adapted from the `official code
    <https://github.com/raminnakhli/Decoupled-Contrastive-Learning>`_
    """

    def __init__(
        self,
        temperature: float = 0.1,
        *,
        weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        """
        :param weight_fn: The weighting function of the positive sample loss.
        :param temperature: Temperature controlling the sharpness of the softmax distribution.
        """
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Calculates the one-way decoupled constrastive loss.

        :param z1: First embedding vector
        :param z2: Second embedding vector
        :return: One-way loss between the embedding vectors.
        """
        return decoupled_constrastive_loss(
            z1, z2, temperature=self.temperature, weight_fn=self.weight_fn
        )

    @classmethod
    def with_vmf_weighting(
        cls: Type[Self], sigma: float = 0.5, *, temperature: float = 0.1
    ) -> Self:
        """
        Initialise the DCL loss with von Mises-Fisher weighting.

        :param sigma: :math:`\\sigma` (scale) parameter for the weigting function.
        :param temperature: Temperature controlling the sharpness of the softmax distribution.
        """
        return cls(temperature=temperature, weight_fn=_von_mises_fisher_weighting)
