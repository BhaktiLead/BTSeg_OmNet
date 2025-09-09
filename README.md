# OMNet — Brain Tumor Segmentation (Rewritten)


This repository contains a complete PyTorch implementation of an OMNet-style segmentation model for the Kaggle Brain Tumor Segmentation dataset located at `/kaggle/input/brain-tumor-segmentation`.


## Files
- `model.py` — OMNet architecture (PyTorch)
- `dataset.py` — Dataset and augmentations
- `train.py` — Training loop with checkpointing & metrics
- `infer.py` — Inference / prediction script
- `utils.py` — Helpers (metrics, save/load)
- `requirements.txt` — Python dependencies
- `config.py` — Configuration values


## How to run (Kaggle)
1. Place this project in a Kaggle notebook (or mount your local environment) with the dataset available at `/kaggle/input/brain-tumor-segmentation`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Train: `python train.py --epochs 50 --batch-size 8 --lr 1e-4`
4. Infer: `python infer.py --checkpoint checkpoints/best.pth --save-dir outputs`


## Notes
- Uses PyTorch. Designed for clarity and reproducibility.
- Modify `config.py` to change paths and hyperparameters.
