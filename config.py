import os


# Paths
DATA_ROOT = os.environ.get('DATA_ROOT', '/kaggle/input/brain-tumor-segmentation')
TRAIN_IMAGES = os.path.join(DATA_ROOT, 'images')
TRAIN_MASKS = os.path.join(DATA_ROOT, 'masks')
CHECKPOINT_DIR = 'checkpoints'
OUTPUT_DIR = 'outputs'


# Training
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
NUM_WORKERS = 2
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50


# Model
IN_CHANNELS = 3
OUT_CHANNELS = 1


# Misc
SEED = 42


# Create dirs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
