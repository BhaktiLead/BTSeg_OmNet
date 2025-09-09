import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim


from config import *
from dataset import BrainTumorDataset, get_training_transforms, get_validation_transforms
from model import OMNet
from utils import dice_coeff, save_checkpoint


# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)




def train_one_epoch(model, loader, criterion, optimizer, device):
model.train()
epoch_loss = 0
for images, masks, _ in tqdm(loader, desc='train', leave=False):
images = images.to(device)
masks = masks.to(device)
preds = model(images)
loss = criterion(preds, masks)
optimizer.zero_grad()
loss.backward()
optimizer.step()
epoch_loss += loss.item()
return epoch_loss / len(loader)




def validate(model, loader, criterion, device):
model.eval()
val_loss = 0
dices = []
with torch.no_grad():
for images, masks, _ in tqdm(loader, desc='val', leave=False):
images = images.to(device)
masks = masks.to(device)
preds = model(images)
loss = criterion(preds, masks)
val_loss += loss.item
