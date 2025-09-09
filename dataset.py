import os
from glob import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BrainTumorDataset(Dataset):
    """
    Custom Dataset for Brain Tumor Segmentation.
    Loads images and corresponding masks from given directories.
    """
    def __init__(self, images_dir, masks_dir, file_list=None, transforms=None):
        """
        Args:
            images_dir (str): Path to image folder.
            masks_dir (str): Path to mask folder.
            file_list (list, optional): List of filenames to use. 
                                        If None, loads all files in images_dir.
            transforms (albumentations.Compose, optional): Data augmentation pipeline.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        if file_list is None:
            self.files = sorted([
                os.path.basename(p) for p in glob(os.path.join(images_dir, '*'))
            ])
        else:
            self.files = file_list

        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Get filename
        fname = self.files[idx]

        # Construct paths
        img_path = os.path.join(self.images_dir, fname)
        mask_path = os.path.join(self.masks_dir, fname)

        # Load image (RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            # If mask missing, create empty mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Normalize mask to 0/1
        mask = (mask > 0).astype('uint8')

        # Apply augmentations / preprocessing
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Basic preprocessing without Albumentations
            image = image.astype("float32") / 255.0
            image = np.transpose(image, (2, 0, 1))   # HWC → CHW
            mask = np.expand_dims(mask.astype("float32"), 0)

        return image, mask, fname


def get_training_transforms(img_size=256):
    """
    Data augmentation for training.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2()
    ])


def get_validation_transforms(img_size=256):
    """
    Preprocessing for validation / test.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])
