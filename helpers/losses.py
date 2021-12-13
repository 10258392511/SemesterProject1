import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_to_one_hot(mask, num_classes):
    # mask: (B, 1, H, W)
    B, _, H, W = mask.shape
    mask_out = torch.zeros((B, num_classes, H, W), dtype=mask.dtype, device=mask.device)
    mask_out.scatter_(1, mask, 1)

    return mask_out


def cross_entropy_loss(X, mask):
    # X: (B, K, H, W), mask: (B, 1, H, W) -> (B, H, W)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.3, 0.3, 0.3]).to(X.device))
    loss = criterion(X, mask.squeeze(1))

    return loss


def dice_loss(X, mask, if_soft_max=True, eps=1e-10):
    # X: (B, K, H, W), mask: (B, 1, H, W)
    num_classes = X.shape[1]
    if if_soft_max:
        X1 = F.softmax(X, dim=1)  # (B, K, H, W)
    else:
        X1 = X
    X1_reduced = X1.sum(dim=[2, 3])  # (B, K)
    mask1 = mask_to_one_hot(mask, num_classes)  # (B, K, H, W)
    X2 = X1 * mask1  # (B, K, H, W)
    X2_reduced = X2.sum(dim=[2, 3])  # (B, K)
    mask_reduced = mask1.sum(dim=[2, 3])
    # remove background
    loss = 1 - ((2 * X2_reduced) / (X1_reduced + mask_reduced + eps)).mean()  # float

    return loss


def symmetric_loss(X1, X2, loss_fn):
    # X1, X2: (B, K, H, W)
    # loss_fn(X, mask): X: (B, K, H, W), mask: (B, 1, H, W)
    mask2 = X2.data
    loss1 = loss_fn(X1, mask2.detach().argmax(dim=1, keepdims=True))
    mask1 = X1.data
    loss2 = loss_fn(X2, mask1.detach().argmax(dim=1, keepdims=True))

    return (loss1 + loss2) / 2
