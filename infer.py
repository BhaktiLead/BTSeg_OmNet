import os
import argparse
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import BrainTumorDataset, get_validation_transforms
from model import OMNet
from utils import load_checkpoint


def inference(model, loader, device, save_dir):
    """
    Run inference on dataset and save predicted masks.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for images, _, fnames in tqdm(loader, desc="inference"):
            images = images.to(device)
            preds = model(images)
            preds = preds.cpu().numpy()

            for pred, fname in zip(preds, fnames):
                # Convert prediction to mask
                mask = (pred[0] > 0.5).astype("uint8") * 255

                # Save output mask
                save_path = os.path.join(save_dir, fname)
                cv2.imwrite(save_path, mask)


def main():
    parser = argparse.ArgumentParser(description="Inference for Brain Tumor Segmentation (OMNet)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pth file)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Image size for preprocessing")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--save-dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save predictions")
    args = parser.parse_args()

    device = DEVICE

    # Collect image filenames
    test_files = sorted([
        os.path.basename(p) for p in __import__('glob').glob(os.path.join(TRAIN_IMAGES, '*'))
    ])

    # Dataset & Loader
    test_ds = BrainTumorDataset(TRAIN_IMAGES, TRAIN_MASKS,
                                file_list=test_files,
                                transforms=get_validation_transforms(args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model = OMNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model)

    # Run inference
    inference(model, test_loader, device, args.save_dir)

    print(f"Inference complete. Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
