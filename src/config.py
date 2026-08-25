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
IS_KAGGLE = os.path.exists("/kaggle")

# ======================================================
# Project Root
# ======================================================

if IS_KAGGLE:

    # GitHub project cloned in Kaggle
    PROJECT_ROOT = Path("/kaggle/working/HandwrittenMathVerifier")

    # HME100K copied to Kaggle working SSD
    DATASET_DIR = Path(
        "/kaggle/working/dataset/"
        "hme100k-handwritten-mathematical-expressions/HME100K"
    )

    # Models saved in the Kaggle project workspace
    MODEL_DIR = PROJECT_ROOT / "saved_models"

elif IS_COLAB:

    # GitHub project cloned in Colab
    PROJECT_ROOT = Path("/content/HandwrittenMathVerifier")

    # Dataset extracted on Colab SSD (FAST)
    DATASET_DIR = Path("/content/HME100K")

    # Save trained models permanently to Google Drive
    MODEL_DIR = Path("/content/drive/MyDrive/HandwrittenMathVerifier/saved_models")

else:

    # Ubuntu / Local
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Local dataset
    DATASET_DIR = PROJECT_ROOT / "dataset"

    # Local models
    MODEL_DIR = PROJECT_ROOT / "saved_models"

# ======================================================
# Dataset Paths
# ======================================================

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
# Outputs
# ======================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ======================================================
# Image Settings
# ======================================================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1536
CHANNELS = 3

# ======================================================
# Training
# ======================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30

NUM_WORKERS = 2 if (IS_COLAB or IS_KAGGLE) else 4

# ======================================================
# Device
# ======================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ======================================================
# Create Directories
# ======================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# Print Information
# ======================================================

print("=" * 60)

if IS_KAGGLE:
    print("Running on Kaggle")
elif IS_COLAB:
    print("Running on Google Colab")
else:
    print("Running on Local Machine")

print("Device  :", DEVICE)
print("Project :", PROJECT_ROOT)
print("Dataset :", DATASET_DIR)
print("Models  :", MODEL_DIR)

print("=" * 60)