import os
from pathlib import Path
import torch

# ======================================================
# Environment Detection
# ======================================================
# IMPORTANT: Colab is checked FIRST. Some environments can
# expose /kaggle/working, so that directory alone must NOT
# mean "Kaggle".
IS_COLAB = (
    os.environ.get("COLAB_RELEASE_TAG") is not None
    or os.environ.get("COLAB_GPU") is not None
    or os.environ.get("COLAB_JUPYTER_TRANSPORT") is not None
    or (
        Path("/content").exists()
        and (
            Path("/content/HandwrittenMathVerifier").exists()
            or Path("/content/HME100K").exists()
            or Path("/content/drive").exists()
        )
    )
)

IS_KAGGLE = (
    not IS_COLAB
    and (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or Path("/kaggle/input").exists()
    )
)

# ======================================================
# Project + Dataset
# ======================================================
if IS_COLAB:
    PROJECT_ROOT = Path("/content/HandwrittenMathVerifier")

    DATASET_CANDIDATES = [
        Path("/content/HME100K"),
        PROJECT_ROOT / "HME100K",
        Path("/content/drive/MyDrive/HandwrittenMathVerifier/HME100K"),
        Path("/content/drive/MyDrive/HME100K"),
    ]

    DATASET_DIR = None
    for candidate in DATASET_CANDIDATES:
        if (
            (candidate / "train" / "train_images").is_dir()
            and (candidate / "train" / "train_labels.txt").is_file()
        ):
            DATASET_DIR = candidate
            break

    if DATASET_DIR is None:
        DATASET_DIR = Path("/content/HME100K")

    DRIVE_ROOT = Path(
        "/content/drive/MyDrive/HandwrittenMathVerifier"
    )

    MODEL_DIR = (
        DRIVE_ROOT / "saved_models"
        if DRIVE_ROOT.exists()
        else PROJECT_ROOT / "saved_models"
    )

elif IS_KAGGLE:
    PROJECT_ROOT = Path(
        "/kaggle/working/HandwrittenMathVerifier"
    )

    DATASET_CANDIDATES = [
        Path("/kaggle/working/HME100K"),
        Path("/kaggle/input/hme100k"),
        Path(
            "/kaggle/input/"
            "hme100k-handwritten-mathematical-expressions/"
            "HME100K"
        ),
        Path(
            "/kaggle/working/dataset/"
            "hme100k-handwritten-mathematical-expressions/"
            "HME100K"
        ),
    ]

    DATASET_DIR = None
    for candidate in DATASET_CANDIDATES:
        if (
            (candidate / "train" / "train_images").is_dir()
            and (candidate / "train" / "train_labels.txt").is_file()
        ):
            DATASET_DIR = candidate
            break

    if DATASET_DIR is None:
        DATASET_DIR = DATASET_CANDIDATES[0]

    MODEL_DIR = PROJECT_ROOT / "saved_models"

else:
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

if not TRAIN_IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Training image directory not found: {TRAIN_IMAGE_DIR}"
    )

if not TRAIN_LABEL_FILE.is_file():
    raise FileNotFoundError(
        f"Training label file not found: {TRAIN_LABEL_FILE}"
    )

# ======================================================
# Config / Output
# ======================================================
CONFIG_DIR = PROJECT_ROOT / "configs"
CHAR2IDX_FILE = CONFIG_DIR / "char2idx.json"
IDX2CHAR_FILE = CONFIG_DIR / "idx2char.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ======================================================
# Image
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
NUM_WORKERS = 2 if IS_COLAB else 4

# ======================================================
# Device
# ======================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(
    "Running on Google Colab"
    if IS_COLAB
    else "Running on Kaggle"
    if IS_KAGGLE
    else "Running on Local Machine"
)
print("Device       :", DEVICE)

if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))

print("Project      :", PROJECT_ROOT)
print("Dataset      :", DATASET_DIR)
print("Train images :", TRAIN_IMAGE_DIR)
print("Train labels :", TRAIN_LABEL_FILE)
print("Models       :", MODEL_DIR)
print("=" * 60)
