# -*- coding: utf-8 -*-
"""Loss functions: BCE + Dice + soft-clDice (topology-aware)."""
import torch
import torch.nn.functional as F


def soft_skeleton(x, iters=6):
    """Differentiable soft skeletonisation via iterative min/max pooling."""
    x1 = -F.max_pool2d(-x, 3, stride=1, padding=1)   # erosion
    for _ in range(iters - 1):
        x1 = -F.max_pool2d(-x1, 3, stride=1, padding=1)
        x1 = torch.clamp(x1 + F.max_pool2d(x1, 3, stride=1, padding=1) - x1, 0, 1)
    sk = torch.clamp(x - x1, 0, 1)
    return sk


def dice_loss(p, y, eps=1e-6):
    inter = (p * y).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def cldice_loss(p, y, eps=1e-6):
    """soft-clDice loss (Shit et al., 2021)."""
    sk_p = soft_skeleton(p)
    sk_y = soft_skeleton(y)
    tprec = ((sk_p * y).sum(dim=(1, 2, 3)) + eps) / (sk_p.sum(dim=(1, 2, 3)) + eps)
    tsens = ((sk_y * p).sum(dim=(1, 2, 3)) + eps) / (sk_y.sum(dim=(1, 2, 3)) + eps)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


def combined_loss(logit, y, lam_cld=0.5):
    """L = 0.5*BCE + L_Dice + lambda * L_clDice"""
    p = torch.sigmoid(logit)
    l = 0.5 * F.binary_cross_entropy_with_logits(logit, y) + dice_loss(p, y)
    if lam_cld > 0:
        l = l + lam_cld * cldice_loss(p, y)
    return l
