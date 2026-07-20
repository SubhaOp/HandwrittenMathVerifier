"""
Global Configuration
Handwritten Mathematical Expression Recognition
Works on:
    - Ubuntu
    - Google Colab
"""

import os
from pathlib import Path
import torch

# ======================================================
# Detect Environment
# ======================================================

IS_COLAB = os.path.exists("/content")

# ======================================================
# Project Root
# ======================================================

if IS_COLAB:

    # Example:
    # /content/HandwrittenMathVerifier
    PROJECT_ROOT = Path("/content/HandwrittenMathVerifier")

    # Google Drive
    DRIVE_ROOT = Path("/content/drive/MyDrive")

else:

    # Ubuntu / Local
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ======================================================
# Dataset Paths
# ======================================================

if IS_COLAB:

    # Store the dataset once in Google Drive.
    #
    # Example:
    # MyDrive/
    #     HME100K/
    #         train/
    #         test/
    #
    DATASET_DIR = DRIVE_ROOT / "HME100K"

else:

    DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "train_images"
TRAIN_LABEL_FILE = DATASET_DIR / "train" / "train_labels.txt"

TEST_IMAGE_DIR = DATASET_DIR / "test" / "test_images"
TEST_LABEL_FILE = DATASET_DIR / "test" / "test_labels.txt"

# ======================================================
# Config Files
# ======================================================

CONFIG_DIR = PROJECT_ROOT / "configs"

CHAR2IDX_FILE = CONFIG_DIR / "char2idx.json"
IDX2CHAR_FILE = CONFIG_DIR / "idx2char.json"

# ======================================================
# Save Models
# ======================================================

if IS_COLAB:

    MODEL_DIR = DRIVE_ROOT / "HandwrittenMathVerifier" / "saved_models"

else:

    MODEL_DIR = PROJECT_ROOT / "saved_models"

# ======================================================
# Outputs
# ======================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ======================================================
# Image Settings
# ======================================================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1536

# RGB Images
CHANNELS = 3

# ======================================================
# Training
# ======================================================

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

EPOCHS = 30

NUM_WORKERS = 2 if IS_COLAB else 4

# ======================================================
# Device
# ======================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ======================================================
# Create Directories
# ======================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ======================================================
# Print Information
# ======================================================

print("=" * 50)

if IS_COLAB:
    print("Running on Google Colab")
else:
    print("Running on Local Machine")

print("Device :", DEVICE)
print("Dataset:", DATASET_DIR)
print("Models :", MODEL_DIR)

print("=" * 50)