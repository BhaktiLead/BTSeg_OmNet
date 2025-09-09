import torch
import numpy as np
from sklearn.metrics import jaccard_score


def dice_coeff(pred, target, eps=1e-7):
    """
    Compute Dice Coefficient.
    Args:
        pred (np.array): Predicted mask (values in [0,1]).
        target (np.array): Ground truth mask (0 or 1).
        eps (float): Small value to avoid division by zero.
    Returns:
        float: Dice score.
    """
    pred = (pred > 0.5).astype("float32")
    target = target.astype("float32")

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def iou_score(pred, target, eps=1e-7):
    """
    Compute Intersection over Union (IoU).
    Args:
        pred (np.array): Predicted mask (values in [0,1]).
        target (np.array): Ground truth mask (0 or 1).
        eps (float): Small value to avoid division by zero.
    Returns:
        float: IoU score.
    """
    pred = (pred > 0.5).astype("uint8").ravel()
    target = target.astype("uint8").ravel()

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0  # Both empty masks

    return jaccard_score(target, pred)


def save_checkpoint(state, filename):
    """
    Save model checkpoint.
    Args:
        state (dict): Dictionary containing model & optimizer states.
        filename (str): Path to save checkpoint.
    """
    torch.save(state, filename)


def load_checkpoint(path, model, optimizer=None):
    """
    Load model checkpoint.
    Args:
        path (str): Path to checkpoint file.
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state.
    Returns:
        dict: Loaded checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    if optimizer and "opt_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt_state"])

    return checkpoint
