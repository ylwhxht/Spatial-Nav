"""Losses for candidate-relative latent spatial rollout.

The functions in this module never predict a navigation action.  They supervise
the bounded relation increment that is later consumed by the native
Transformer/LM-head decision path.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def fixed_orthonormal_projection(
    input_dim: int,
    output_dim: int,
    *,
    seed: int,
) -> Tensor:
    """Build a deterministic, action-blind teacher coordinate projection.

    Rows are orthonormal, so this is a fixed change of supervision coordinates
    rather than a learned auxiliary model.  It is used only while training;
    runtime does not load or evaluate it.
    """

    if input_dim < 1 or output_dim < 1 or output_dim > input_dim:
        raise ValueError("projection dimensions must satisfy 1 <= out <= in")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_basis = torch.randn(
        input_dim, output_dim, generator=generator, dtype=torch.float32
    )
    basis = torch.linalg.qr(random_basis, mode="reduced").Q
    return basis.transpose(0, 1).contiguous()


def candidate_center(
    values: Tensor,
    active: Tensor,
) -> Tensor:
    """Remove the within-image, same-ability candidate-common component."""

    if values.ndim != 4 or active.shape != values.shape[:3]:
        raise ValueError(
            "candidate centering requires values [B,C,A,H] and active [B,C,A]"
        )
    weights = active.to(values.dtype)[..., None]
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    center = (values * weights).sum(dim=1, keepdim=True) / denominator
    return (values - center) * weights


def relation_delta_contrast_loss(
    relation_delta_hidden: Tensor,
    candidate_contrast_targets: Tensor,
    active: Tensor,
    *,
    negative_margin: float = 0.10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Align only learned relation deltas to action-blind candidate contrasts.

    A candidate-common offset is removed per image and ability.  Consequently,
    the objective cannot ask the relation increment to reconstruct the
    candidate-local visual base.  The hard negative is another public
    candidate from the same image and never uses the policy action.
    """

    if (
        relation_delta_hidden.ndim != 4
        or candidate_contrast_targets.shape != relation_delta_hidden.shape
        or active.shape != relation_delta_hidden.shape[:3]
    ):
        raise ValueError(
            "delta contrast requires matching [B,C,A,H] tensors and "
            "[B,C,A] activity"
        )
    if negative_margin < 0:
        raise ValueError("negative_margin must be non-negative")
    valid = active.bool()
    predicted = F.normalize(
        candidate_center(relation_delta_hidden.float(), valid),
        dim=-1,
    )
    target = F.normalize(
        candidate_center(candidate_contrast_targets.float(), valid),
        dim=-1,
    )
    positive = (predicted * target).sum(-1)
    if not valid.any():
        return predicted.sum() * 0.0, positive.detach(), valid
    positive_loss = (1.0 - positive[valid]).mean()

    negative_target = target.roll(1, dims=1)
    negative_valid = valid & valid.roll(1, dims=1)
    negative = (predicted * negative_target).sum(-1)
    if negative_valid.any():
        ranking = F.relu(
            float(negative_margin)
            - positive[negative_valid]
            + negative[negative_valid]
        ).mean()
    else:
        ranking = positive_loss * 0.0
    return positive_loss + ranking, positive.detach(), valid
