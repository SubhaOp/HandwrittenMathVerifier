# Configuration placeholder
"""
Global configuration for Handwritten Math Verifier
"""

from pathlib import Path
import torch

# ======================================================
# Project Paths
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "saved_models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ======================================================
# Dataset
# ======================================================

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "train_images"
TRAIN_LABEL_FILE = DATASET_DIR / "train" / "train_labels.txt"

TEST_IMAGE_DIR = DATASET_DIR / "test" / "test_images"
TEST_LABEL_FILE = DATASET_DIR / "test" / "test_labels.txt"

# ======================================================
# Vocabulary
# ======================================================

CHAR2IDX_FILE = CONFIG_DIR / "char2idx.json"
IDX2CHAR_FILE = CONFIG_DIR / "idx2char.json"

# ======================================================
# Image
# ======================================================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1536
CHANNELS = 1

# ======================================================
# Training
# ======================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 30
NUM_WORKERS = 2

# ======================================================
# Model
# ======================================================

EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1

# ======================================================
# Device
# ======================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Create folders if missing
# ======================================================

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)