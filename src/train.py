"""
Global Configuration
Handwritten Mathematical Expression Recognition

Environment-aware configuration for:
- Google Colab
- Kaggle
- Local Windows/Linux

The dataset path is detected automatically from known locations.
"""

import os
from pathlib import Path
import torch


# ======================================================
# Detect Environment
# ======================================================

IS_COLAB = os.path.exists("/content") and (
    "COLAB_RELEASE_TAG" in os.environ
    or os.path.exists("/content/drive")
)

IS_KAGGLE = os.path.exists("/kaggle/working")


# ======================================================
# Project Root + Dataset
# ======================================================

if IS_COLAB:

    PROJECT_ROOT = Path("/content/HandwrittenMathVerifier")

    # Possible HME100K locations in Colab.
    DATASET_CANDIDATES = [
        Path("/content/HME100K"),
        PROJECT_ROOT / "HME100K",
        Path("/content/drive/MyDrive/HandwrittenMathVerifier/HME100K"),
        Path("/content/drive/MyDrive/HME100K"),
    ]

    DATASET_DIR = None

    for candidate in DATASET_CANDIDATES:
        if (
            (candidate / "train" / "train_images").exists()
            and (candidate / "train" / "train_labels.txt").exists()
        ):
            DATASET_DIR = candidate
            break

    if DATASET_DIR is None:
        # Default expected location. A clear error is raised below.
        DATASET_DIR = Path("/content/HME100K")

    # Permanent model/epoch-study storage.
    DRIVE_MODEL_DIR = Path(
        "/content/drive/MyDrive/HandwrittenMathVerifier/saved_models"
    )

    if Path("/content/drive/MyDrive").exists():
        MODEL_DIR = DRIVE_MODEL_DIR
    else:
        MODEL_DIR = PROJECT_ROOT / "saved_models"


elif IS_KAGGLE:

    PROJECT_ROOT = Path("/kaggle/working/HandwrittenMathVerifier")

    DATASET_CANDIDATES = [
        Path("/kaggle/working/HME100K"),
        Path("/kaggle/input/hme100k"),
        Path("/kaggle/input/hme100k-handwritten-mathematical-expressions/HME100K"),
        Path("/kaggle/working/dataset/hme100k-handwritten-mathematical-expressions/HME100K"),
    ]

    DATASET_DIR = None

    for candidate in DATASET_CANDIDATES:
        if (
            (candidate / "train" / "train_images").exists()
            and (candidate / "train" / "train_labels.txt").exists()
        ):
            DATASET_DIR = candidate
            break

    if DATASET_DIR is None:
        DATASET_DIR = DATASET_CANDIDATES[0]

    MODEL_DIR = PROJECT_ROOT / "saved_models"

else:

    # Local Windows/Linux
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_DIR = PROJECT_ROOT / "dataset"
    MODEL_DIR = PROJECT_ROOT / "saved_models"


# ======================================================
# Dataset Paths
# ======================================================

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "train_images"
TRAIN_LABEL_FILE = DATASET_DIR / "train" / "train_labels.txt"

TEST_IMAGE_DIR = DATASET_DIR / "test" / "test_images"
TEST_LABEL_FILE = DATASET_DIR / "test" / "test_labels.txt"


# ======================================================
# Validate Dataset Path
# ======================================================

if not TRAIN_IMAGE_DIR.exists() or not TRAIN_LABEL_FILE.exists():

    raise FileNotFoundError(
        "\n\nHME100K DATASET NOT FOUND.\n"
        f"Expected dataset directory:\n  {DATASET_DIR}\n\n"
        "The following locations were checked:\n"
        + "\n".join(
            f"  - {p}"
            for p in (
                DATASET_CANDIDATES
                if "DATASET_CANDIDATES" in globals()
                else [DATASET_DIR]
            )
        )
        + "\n\n"
        "For Colab, make sure the dataset is available at:\n"
        "  /content/HME100K/train/train_images\n"
        "  /content/HME100K/train/train_labels.txt\n"
    )


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

# train.py controls the epoch-study ceiling.
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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# Print Information
# ======================================================

print("=" * 60)

if IS_COLAB:
    print("Running on Google Colab")
elif IS_KAGGLE:
    print("Running on Kaggle")
else:
    print("Running on Local Machine")

print("Device  :", DEVICE)

if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))

print("Project :", PROJECT_ROOT)
print("Dataset :", DATASET_DIR)
print("Train images :", TRAIN_IMAGE_DIR)
print("Train labels :", TRAIN_LABEL_FILE)
print("Models  :", MODEL_DIR)

print("=" * 60)
