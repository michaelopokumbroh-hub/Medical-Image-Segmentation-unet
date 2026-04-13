import numpy as np
import torch

def dice_score_numpy(pred, target, threshold=0.5, eps=1e-7):
    pred = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2. * intersection + eps) / (union + eps)

def dice_torch(preds, targets, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    return ((2*inter + eps) / (union + eps)).mean().item()

def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
