import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from dataset import BrainTumorDataset
from model import OMNet
from utils import dice_coef, save_checkpoint


def train():
    # Load configuration
    cfg = Config()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    train_dataset = BrainTumorDataset(cfg.train_images, cfg.train_masks, augment=True)
    val_dataset = BrainTumorDataset(cfg.val_images, cfg.val_masks, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Model, loss, optimizer
    model = OMNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_dice = 0.0

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{cfg.epochs}] - Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        dice_score = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                dice_score += dice_coef(preds, masks).item()

        dice_score /= len(val_loader)
        print(f"Validation Dice: {dice_score:.4f}")

        # Save best model
        if dice_score > best_dice:
            best_dice = dice_score
            save_checkpoint(model, optimizer, epoch, dice_score, cfg.checkpoint_path)
            print(f"✅ Saved new best model at epoch {epoch+1} with Dice {dice_score:.4f}")


if __name__ == "__main__":
    train()
