"""RMI loss utils.

# Code adapted from:
# https://github.com/ZJULearning/RMI
"""
from typing import Tuple

import torch


def map_get_pairs(
    labels_4d: torch.Tensor, probs_4d: torch.Tensor, radius: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get map pairs."""
    # the original height and width
    label_shape = labels_4d.size()
    h, w = label_shape[2], label_shape[3]
    new_h, new_w = h - (radius - 1), w - (radius - 1)

    # get the neighbors
    la_ns = []
    pr_ns = []
    # for x in range(0, radius, 1):
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            la_now = labels_4d[:, :, y : y + new_h, x : x + new_w]
            pr_now = probs_4d[:, :, y : y + new_h, x : x + new_w]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    la_vectors = torch.stack(la_ns, dim=2)
    pr_vectors = torch.stack(pr_ns, dim=2)

    return la_vectors, pr_vectors


def log_det_by_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Compute log det by cholesky.

    Args:
        matrix (torch.Tensor): matrix must be a positive define matrix.
        shape [N, C, D, D].
    Ref:
        https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py # pylint: disable=line-too-long
    """
    # This uses the property that the log det(A) = 2 * sum(log(real(diag(C))))
    # where C is the cholesky decomposition of A.
    chol = torch.linalg.cholesky(matrix)

    return 2.0 * torch.sum(
        torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )
