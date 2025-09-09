import torch
import numpy as np
from sklearn.metrics import jaccard_score


def dice_coeff(pred, target, eps=1e-7):
pred = (pred > 0.5).astype('float32')
target = target.astype('float32')
intersection = (pred * target).sum()
union = pred.sum() + target.sum()
dice = (2. * intersection + eps) / (union + eps)
return dice


def iou_score(pred, target, eps=1e-7):
pred = (pred > 0.5).astype('uint8').ravel()
target = target.astype('uint8').ravel()
if pred.sum() == 0 and target.sum() == 0:
return 1.0
return jaccard_score(target, pred)


def save_checkpoint(state, filename):
torch.save(state, filename)


def load_checkpoint(path, model, optimizer=None):
checkpoint = torch.load(path, map_location='cpu')
model.load_state_dict(checkpoint['model_state'])
if optimizer and 'opt_state' in checkpoint:
optimizer.load_state_dict(checkpoint['opt_state'])
return checkpoint
