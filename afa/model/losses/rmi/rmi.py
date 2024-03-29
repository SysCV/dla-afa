"""RMI loss.

# Code adapted from:
# https://github.com/ZJULearning/RMI
"""
import torch
import torch.nn.functional as F
from torch import nn

from .rmi_utils import log_det_by_cholesky, map_get_pairs

# min clip value after softmax or sigmoid operations
CLP_MIN = 1e-6
# add this factor to ensure the AA^T is positive definite
POS_ALPHA = 5e-4
# sum the loss per channel
IS_SUM = 1


class RMILoss(nn.Module):  # type: ignore
    """Region mutual information loss.

    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    def __init__(
        self,
        num_classes: int,
        rmi_radius: int = 3,
        rmi_pool_way: int = 1,
        rmi_pool_size: int = 4,
        rmi_pool_stride: int = 4,
        loss_weight_lambda: float = 0.5,
        lambda_way: int = 1,
        ignore_index: int = 255,
        pos_weight: float = 1.0,
        is_edge: bool = False,
    ) -> None:
        """Init."""
        super().__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius

        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = ignore_index
        # pos weight
        self.pos_weight = pos_weight
        self.is_edge = is_edge

    def forward(
        self,
        logits_4d: torch.Tensor,
        labels_4d: torch.Tensor,
        do_rmi: bool = True,
    ) -> torch.Tensor:
        """Forward.

        Args:
            logits_4d: [N, C, H, W]
            labels_4d: [N, H, W]
            do_rmi: whether do rmi
        """
        logits_4d.float()
        labels_4d.float()
        # label mask -- [N, H, W, 1]
        label_mask_3d = labels_4d < self.num_classes

        # valid label
        valid_onehot_labels_4d = F.one_hot(
            labels_4d.long() * label_mask_3d.long(),
            num_classes=self.num_classes,
        ).float()
        label_mask_3d = label_mask_3d.float()
        label_mask_flat = label_mask_3d.view(
            [
                -1,
            ]
        )
        valid_onehot_labels_4d = (
            valid_onehot_labels_4d * label_mask_3d.unsqueeze(dim=3)
        )
        valid_onehot_labels_4d.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4d.view(
            [-1, self.num_classes]
        ).requires_grad_(False)
        logits_flat = (
            logits_4d.permute(0, 2, 3, 1)
            .contiguous()
            .view([-1, self.num_classes])
        )

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        if self.pos_weight != 1.0:
            pos_weight = logits_flat.new_tensor([1.0, self.pos_weight])
        else:
            pos_weight = None
        binary_loss = F.binary_cross_entropy_with_logits(
            logits_flat,
            target=valid_onehot_label_flat,
            weight=label_mask_flat.unsqueeze(dim=1),
            reduction="sum",
            pos_weight=pos_weight,
        )
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)

        if not do_rmi or self.is_edge:
            final_loss = bce_loss
        else:
            # PART II -- get rmi loss
            # onehot_labels_4d -- [N, C, H, W]
            probs_4d = (
                logits_4d.sigmoid() * label_mask_3d.unsqueeze(dim=1) + CLP_MIN
            )
            valid_onehot_labels_4d = valid_onehot_labels_4d.permute(
                0, 3, 1, 2
            ).requires_grad_(False)

            # get region mutual information
            rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4d, probs_4d)

            # add together
            if self.lambda_way:
                final_loss = self.weight_lambda * bce_loss + rmi_loss * (
                    1 - self.weight_lambda
                )
            else:
                final_loss = bce_loss + rmi_loss * self.weight_lambda

        return final_loss

    @staticmethod
    def inverse(input_x: torch.Tensor) -> torch.Tensor:
        """Inverse."""
        return torch.inverse(input_x)

    def rmi_lower_bound(
        self, labels_4d: torch.Tensor, probs_4d: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the lower bound of the region mutual information.

        Args:
            labels_4d (float32): [N, C, H, W]
            probs_4d (float32): [N, C, H, W]
        :raises NotImplementedError:
        """
        assert labels_4d.size() == probs_4d.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4d = F.max_pool2d(
                    labels_4d,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding,
                )
                probs_4d = F.max_pool2d(
                    probs_4d,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding,
                )
            elif self.rmi_pool_way == 1:
                labels_4d = F.avg_pool2d(
                    labels_4d,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding,
                )
                probs_4d = F.avg_pool2d(
                    probs_4d,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding,
                )
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4d.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4d = F.interpolate(
                    labels_4d, size=(new_h, new_w), mode="nearest"
                )
                probs_4d = F.interpolate(
                    probs_4d,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4d.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new
        # shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = map_get_pairs(
            labels_4d, probs_4d, radius=self.rmi_radius
        )

        la_vectors = (
            la_vectors.view([n, c, self.half_d, -1])
            .type(torch.cuda.DoubleTensor)
            .requires_grad_(False)
        )
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(
            torch.cuda.DoubleTensor
        )

        # small diagonal matrix,
        # shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        # pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) *
        # POS_ALPHA)
        pr_cov_inv = self.inverse(
            pr_cov + diag_matrix.type_as(pr_cov) * POS_ALPHA
        )
        # if the dimension of the point is less than 9, you can use the below
        # function to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov +
        # diag_matrix.type_as(pr_cov) * POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n
        # x n shape; then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by
        # number of points here, and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(
            la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1)
        )
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv,
        # la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) +
        # diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * log_det_by_cholesky(
            appro_var + diag_matrix.type_as(appro_var) * POS_ALPHA
        )
        # rmi_now = 0.5 * torch.logdet(appro_var +
        # diag_matrix.type_as(appro_var) * POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = (
            rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        )
        # is_half = False
        # if is_half:
        # 	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = (
            torch.sum(rmi_per_class) if IS_SUM else torch.mean(rmi_per_class)
        )
        return rmi_loss
