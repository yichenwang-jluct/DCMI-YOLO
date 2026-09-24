# -*- coding: utf-8 -*-
"""
DCMI-YOLO training losses, written from the equations in the manuscript.

    DF-CIoU          Section III-D, Eq. (13)-(15)   - replaces CIoU in the box loss
    Repulsion Loss   Section III-C, Eq. (9)-(12)    - added to the total objective

Everything here is fully specified by the paper, including the constants:

    d0 = 0.2, u0 = 0.8                              Eq. (13)
    lambda_RepGT = lambda_RepBox = sigma = 0.5      Section III-C, grid-searched

The two belong together: Section III-C states that the "MultiSEAM" ablation
column denotes the feature-plus-loss pair, so the reported gain covers the
attention module AND the repulsion terms.
"""

from __future__ import annotations

import math

import torch

__all__ = ["df_ciou", "DFCIoU", "repulsion_loss", "smooth_ln"]


# ---------------------------------------------------------------------------
# Section III-D : Dynamic Focaler-CIoU
# ---------------------------------------------------------------------------
def df_interval(epoch: int, epochs: int, d0: float = 0.2, u0: float = 0.8):
    """
    The mapping interval of Eq. (13), which narrows as training proceeds:

        d(t) = d0 * t/T
        u(t) = 1 - (1 - u0) * t/T

    At t = 0 it is [0, 1], which makes DF-CIoU identical to plain CIoU and keeps
    early convergence safe; by t = T it has tightened to [d0, u0] = [0.2, 0.8].
    """
    r = 0.0 if epochs <= 0 else min(max(epoch / epochs, 0.0), 1.0)
    return d0 * r, 1.0 - (1.0 - u0) * r


def df_ciou(box1: torch.Tensor, box2: torch.Tensor, epoch: int = 0, epochs: int = 200,
            d0: float = 0.2, u0: float = 0.8, xywh: bool = False, eps: float = 1e-7):
    """
    Dynamic Focaler-CIoU, Eq. (14)-(15). Returns the quantity that the caller
    should use exactly where `bbox_iou(..., CIoU=True)` was used, i.e. the box
    loss is `(1 - df_ciou(...)) * weight`.

        IoU_DF = 0                                   IoU <  d(t)
                 (IoU - d) / (u - d)                  d <= IoU <= u             Eq. (14)
                 1                                   IoU >  u

        L = 1 - IoU_DF + rho^2(b, b_gt)/c^2 + alpha*v                           Eq. (15)

    so this function returns  IoU_DF - rho^2/c^2 - alpha*v, whose complement is L.

    Late in training, positives inside the interval get an IoU gradient amplified
    by 1/(u0 - d0) = 1/0.6, while positives outside it are refined only by the
    geometric terms and the DFL loss.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1x1, b1x2, b1y1, b1y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2x1, b2x2, b2y1, b2y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1x1, b1y1, b1x2, b1y2 = box1.chunk(4, -1)
        b2x1, b2y1, b2x2, b2y2 = box2.chunk(4, -1)
        w1, h1 = b1x2 - b1x1, (b1y2 - b1y1).clamp_(eps)
        w2, h2 = b2x2 - b2x1, (b2y2 - b2y1).clamp_(eps)

    inter = (b1x2.minimum(b2x2) - b1x1.maximum(b2x1)).clamp_(0) * \
            (b1y2.minimum(b2y2) - b1y1.maximum(b2y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # --- Eq. (14): linear interval mapping, clamped outside [d, u]
    d, u = df_interval(epoch, epochs, d0, u0)
    iou_df = ((iou - d) / max(u - d, eps)).clamp(0.0, 1.0)

    # --- Eq. (15): the CIoU geometric terms, unchanged from the original definition
    cw = b1x2.maximum(b2x2) - b1x1.minimum(b2x1)
    ch = b1y2.maximum(b2y2) - b1y1.minimum(b2y1)
    c2 = cw.pow(2) + ch.pow(2) + eps                               # enclosing diagonal^2
    rho2 = ((b2x1 + b2x2 - b1x1 - b1x2).pow(2) +
            (b2y1 + b2y2 - b1y1 - b1y2).pow(2)) / 4                # centre distance^2
    v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))                          # CIoU trade-off weight
    return iou_df - (rho2 / c2 + v * alpha)


class DFCIoU:
    """
    Thin holder so the current epoch can reach the loss. Ultralytics' trainer does
    not pass the epoch into the criterion, so set it once per epoch from a callback:

        model.add_callback("on_train_epoch_start",
                           lambda tr: setattr(tr.model.criterion.bbox_loss.dfciou,
                                              "epoch", tr.epoch))
    """

    def __init__(self, epochs: int = 200, d0: float = 0.2, u0: float = 0.8):
        self.epochs, self.d0, self.u0, self.epoch = epochs, d0, u0, 0

    def __call__(self, box1, box2, xywh=False):
        return df_ciou(box1, box2, self.epoch, self.epochs, self.d0, self.u0, xywh)


# ---------------------------------------------------------------------------
# Section III-C : Repulsion Loss
# ---------------------------------------------------------------------------
def smooth_ln(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """
    Eq. (10): continuously differentiable logarithmic smoothing.

        SmoothLn(x) = -ln(1 - x)                        x <= sigma
                      (x - sigma)/(1 - sigma) - ln(1 - sigma)   x >  sigma

    sigma in [0, 1) tunes sensitivity to outliers; the paper's grid search picks 0.5.
    """
    x = x.clamp(0.0, 1.0 - 1e-6)
    lo = -torch.log1p(-x)
    hi = (x - sigma) / (1.0 - sigma) - math.log(1.0 - sigma)
    return torch.where(x <= sigma, lo, hi)


def _iog(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoG = Area(P and G) / Area(G), Eq. (9). Boxes are xyxy, shape (N, 4)."""
    inter = (pred[:, 2].minimum(gt[:, 2]) - pred[:, 0].maximum(gt[:, 0])).clamp_(0) * \
            (pred[:, 3].minimum(gt[:, 3]) - pred[:, 1].maximum(gt[:, 1])).clamp_(0)
    area_g = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).clamp_(eps)
    return inter / area_g


def _pairwise_iou(b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU between every pair of boxes in b (N, 4) -> (N, N)."""
    tl = torch.max(b[:, None, :2], b[None, :, :2])
    br = torch.min(b[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp_(0).prod(-1)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area[:, None] + area[None, :] - inter + eps)


def repulsion_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                   target_idx: torch.Tensor, all_gt: torch.Tensor,
                   sigma: float = 0.5, eps: float = 1e-7):
    """
    Eq. (9) and (11). Returns (L_RepGT, L_RepBox).

    CALL THIS ONCE PER IMAGE, NOT ONCE PER BATCH. Ultralytics' `fg_mask` selects
    positives across the whole batch; feeding those in together would make boxes
    from different images repel each other, which is not what Eq. (9) and (11)
    mean, and would build a P x P matrix far larger than necessary. Loop over the
    batch index and average the per-image results.

    Args:
        pred_boxes   (P, 4)  xyxy, this image's positive predictions
        target_boxes (P, 4)  xyxy, each positive's own assigned ground truth
        target_idx   (P,)    int64 index into `all_gt` of each positive's own target
        all_gt       (G, 4)  xyxy, every ground-truth box in this image
        sigma        SmoothLn threshold; the paper uses 0.5

    L_RepGT pushes each prediction away from the NON-target ground truth it
    overlaps most. L_RepBox pushes apart predictions assigned to different
    targets, normalised by the number of pairs with non-zero overlap.
    """
    device = pred_boxes.device
    zero = torch.zeros((), device=device)
    P, G = pred_boxes.shape[0], all_gt.shape[0]
    if P == 0 or G == 0:
        return zero, zero

    # ---- RepGT: the highest-IoU ground truth that is NOT this prediction's own
    tl = torch.max(pred_boxes[:, None, :2], all_gt[None, :, :2])
    br = torch.min(pred_boxes[:, None, 2:], all_gt[None, :, 2:])
    inter = (br - tl).clamp_(0).prod(-1)
    ap = ((pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]))[:, None]
    ag = ((all_gt[:, 2] - all_gt[:, 0]) * (all_gt[:, 3] - all_gt[:, 1]))[None, :]
    iou_pg = inter / (ap + ag - inter + eps)
    iou_pg.scatter_(1, target_idx.view(-1, 1), -1.0)               # mask out the own target
    best = iou_pg.argmax(1)
    valid = iou_pg.gather(1, best.view(-1, 1)).squeeze(1) > 0
    if valid.any():
        l_repgt = smooth_ln(_iog(pred_boxes[valid], all_gt[best[valid]]), sigma).mean()
    else:
        l_repgt = zero

    # ---- RepBox: predictions belonging to different targets repel each other
    if P > 1:
        iou_pp = _pairwise_iou(pred_boxes)
        diff = target_idx[:, None] != target_idx[None, :]
        keep = torch.triu(diff, diagonal=1) & (iou_pp > 0)
        n = keep.sum()
        l_repbox = smooth_ln(iou_pp[keep], sigma).sum() / (n + eps) if n > 0 else zero
    else:
        l_repbox = zero

    return l_repgt, l_repbox


# ---------------------------------------------------------------------------
# Eq. (12): the total objective
# ---------------------------------------------------------------------------
LAMBDA_REPGT = 0.5      # grid-searched on the in-house validation split
LAMBDA_REPBOX = 0.5
SIGMA = 0.5

def total_loss(l_cls, l_box, l_dfl, l_repgt, l_repbox,
               lam_gt: float = LAMBDA_REPGT, lam_box: float = LAMBDA_REPBOX):
    """L_total = L_cls + L_box + L_dfl + lambda_RepGT * L_RepGT + lambda_RepBox * L_RepBox."""
    return l_cls + l_box + l_dfl + lam_gt * l_repgt + lam_box * l_repbox
