"""
V3 Epoch Study Training
Handwritten Mathematical Expression Recognition

Pipeline:
    HME100K
      -> preprocessing
      -> ResNet18
      -> 384-step sequence
      -> 2-layer BiLSTM
      -> Linear
      -> CTC

Features:
- Google Colab + Google Drive checkpoint storage
- Automatic resume after Colab/runtime interruption
- Checkpoint after every completed epoch
- Permanent checkpoint for every epoch
- Best validation-loss model
- CSV history
- Training/validation loss graph
- Early stopping
- 100-epoch safety ceiling

IMPORTANT:
    src/config.py contains configuration/path settings.
    This file must be src/train.py.
"""

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from src.config import *
from src.dataset import HMEDataset
from src.model import MathRecognizer


# ======================================================
# V3 Epoch Study Configuration
# ======================================================

VALIDATION_RATIO = 0.10
RANDOM_SEED = 42

# 100 is ONLY a safety ceiling.
# Early stopping determines the actual required epoch count.
MAX_EPOCHS = 100
EPOCHS = MAX_EPOCHS

# False = resume from Google Drive checkpoint if present.
# True  = intentionally start a brand-new epoch study.
RESET_EPOCH_STUDY = False

EARLY_STOPPING_PATIENCE = 8

# V3 model is expected to produce 384 CTC time steps.
CTC_TIME_STEPS = 384


# ======================================================
# Reproducibility
# ======================================================

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


# ======================================================
# Device
# ======================================================

print("\n" + "=" * 60)
print("V3 EPOCH STUDY")
print("=" * 60)
print("Device :", DEVICE)

if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
    print("VRAM   :", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("=" * 60)


# ======================================================
# Collate
# ======================================================

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    images = torch.stack(images)

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=0
    )

    return images, labels


# ======================================================
# Target lengths
# ======================================================

def get_target_lengths(labels):
    """
    Padding and CTC blank ID are both 0.
    Actual characters use non-zero IDs.
    """
    return torch.tensor(
        [
            torch.count_nonzero(label).item()
            for label in labels
        ],
        dtype=torch.long,
        device=labels.device
    )


# ======================================================
# Checkpoint helper
# ======================================================

def atomic_torch_save(obj, path):
    """
    Save to a temporary file first, then replace the target.

    This reduces the chance of leaving a partially written
    checkpoint if the runtime is interrupted during saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(obj, temp_path)
    temp_path.replace(path)


# ======================================================
# Dataset
# ======================================================

print("\nLoading Dataset...")

dataset = HMEDataset()

print(f"Original Dataset Samples : {len(dataset)}")


# ======================================================
# CTC Validity Filtering
# ======================================================

print("\nChecking CTC validity...")

valid_indices = []
invalid_indices = []

for idx in tqdm(
    range(len(dataset)),
    desc="Checking labels"
):
    row = dataset.df.iloc[idx]
    label = dataset.encode_label(row["label"])

    target_length = len(label)

    if target_length > 1:
        repeats = int(
            (
                label[1:] == label[:-1]
            ).sum().item()
        )
    else:
        repeats = 0

    required_steps = target_length + repeats

    if required_steps <= CTC_TIME_STEPS:
        valid_indices.append(idx)
    else:
        invalid_indices.append(idx)

print("\n" + "=" * 60)
print("V3 CTC Dataset Filtering")
print("=" * 60)
print(f"Original Samples : {len(dataset)}")
print(f"Valid Samples    : {len(valid_indices)}")
print(f"Filtered Samples : {len(invalid_indices)}")
print("=" * 60)


# ======================================================
# Train / Validation Split
# ======================================================

generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)

permutation = torch.randperm(
    len(valid_indices),
    generator=generator
).tolist()

validation_size = max(
    1,
    int(len(valid_indices) * VALIDATION_RATIO)
)

val_positions = permutation[:validation_size]
train_positions = permutation[validation_size:]

train_indices = [
    valid_indices[i]
    for i in train_positions
]

val_indices = [
    valid_indices[i]
    for i in val_positions
]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

print("\n" + "=" * 60)
print("V3 Dataset Split")
print("=" * 60)
print(f"Training Samples   : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")
print("=" * 60)


# ======================================================
# DataLoaders
# ======================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
    persistent_workers=(NUM_WORKERS > 0)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
    persistent_workers=(NUM_WORKERS > 0)
)


# ======================================================
# Model
# ======================================================

num_classes = len(dataset.char2idx) + 1

model = MathRecognizer(num_classes).to(DEVICE)

print("\nModel Created Successfully")
print(f"Number of Classes : {num_classes}")


# ======================================================
# Verify V3 sequence length
# ======================================================

print("\nChecking V3 model output shape...")

with torch.no_grad():
    test_input = torch.zeros(
        1,
        CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        device=DEVICE
    )

    test_output = model(test_input)

print(f"Input Shape  : {tuple(test_input.shape)}")
print(f"Output Shape : {tuple(test_output.shape)}")

if test_output.shape[1] != CTC_TIME_STEPS:
    raise RuntimeError(
        "V3 sequence length mismatch! "
        f"Expected {CTC_TIME_STEPS}, "
        f"got {test_output.shape[1]}"
    )

print("V3 sequence length verified.")

del test_input
del test_output

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================
# Loss / Optimizer / Scheduler
# ======================================================

criterion = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# ======================================================
# Google Drive Epoch-Study Storage
# ======================================================

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/HandwrittenMathVerifier"
)

if DRIVE_ROOT.exists():
    EPOCH_STUDY_DIR = DRIVE_ROOT / "epoch_study"
    print("\nStorage mode : GOOGLE DRIVE")
else:
    # This fallback keeps the script usable locally/Kaggle.
    if Path("/kaggle/working").exists():
        EPOCH_STUDY_DIR = (
            Path("/kaggle/working")
            / "HandwrittenMathVerifier"
            / "epoch_study"
        )
        print("\nStorage mode : KAGGLE")
    else:
        EPOCH_STUDY_DIR = MODEL_DIR / "epoch_study"
        print("\nStorage mode : LOCAL")

CHECKPOINT_DIR = EPOCH_STUDY_DIR / "checkpoints"

EPOCH_STUDY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR / "latest_checkpoint.pth"
)

HISTORY_CSV = (
    EPOCH_STUDY_DIR / "v3_epoch_history.csv"
)

HISTORY_PLOT = (
    EPOCH_STUDY_DIR / "v3_epoch_analysis.png"
)

BEST_MODEL_PATH = (
    EPOCH_STUDY_DIR / "best_model_v3_epoch_study.pth"
)

FINAL_MODEL_PATH = (
    EPOCH_STUDY_DIR / "math_recognizer_v3_epoch_study.pth"
)


# ======================================================
# History
# ======================================================

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "learning_rate": [],
}


# ======================================================
# Training
# ======================================================

def train_one_epoch():
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=True
    )

    for images, labels in progress_bar:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        outputs = outputs.log_softmax(dim=2)

        # B,T,C -> T,B,C
        outputs = outputs.permute(1, 0, 2)

        input_lengths = torch.full(
            (images.size(0),),
            outputs.size(0),
            dtype=torch.long,
            device=DEVICE
        )

        target_lengths = get_target_lengths(labels)

        loss = criterion(
            outputs,
            labels,
            input_lengths,
            target_lengths
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite CTC loss encountered: {loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    return total_loss / len(train_loader)


# ======================================================
# Validation
# ======================================================

@torch.no_grad()
def validate():
    model.eval()

    total_loss = 0.0

    progress_bar = tqdm(
        val_loader,
        desc="Validation",
        leave=True
    )

    for images, labels in progress_bar:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        outputs = outputs.log_softmax(dim=2)

        outputs = outputs.permute(1, 0, 2)

        input_lengths = torch.full(
            (images.size(0),),
            outputs.size(0),
            dtype=torch.long,
            device=DEVICE
        )

        target_lengths = get_target_lengths(labels)

        loss = criterion(
            outputs,
            labels,
            input_lengths,
            target_lengths
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation CTC loss: {loss.item()}"
            )

        total_loss += loss.item()

        progress_bar.set_postfix(
            val_loss=f"{loss.item():.4f}"
        )

    return total_loss / len(val_loader)


# ======================================================
# Resume Logic
# ======================================================

start_epoch = 0
best_val_loss = float("inf")
best_epoch = 0
no_improvement_epochs = 0
resume_checkpoint = None

if not RESET_EPOCH_STUDY:

    if LATEST_CHECKPOINT.exists():
        resume_checkpoint = LATEST_CHECKPOINT

    else:
        epoch_checkpoints = list(
            CHECKPOINT_DIR.glob(
                "checkpoint_epoch_*.pth"
            )
        )

        if epoch_checkpoints:

            def get_epoch_number(path):
                try:
                    return int(
                        path.stem.split("_")[-1]
                    )
                except ValueError:
                    return -1

            epoch_checkpoints.sort(
                key=get_epoch_number
            )

            resume_checkpoint = (
                epoch_checkpoints[-1]
            )


if RESET_EPOCH_STUDY:

    print("\n" + "=" * 60)
    print("EPOCH STUDY: STARTING FROM SCRATCH")
    print("=" * 60)

elif resume_checkpoint is not None:

    print("\n" + "=" * 60)
    print("RESUMING EPOCH STUDY")
    print("=" * 60)
    print(
        f"Checkpoint : {resume_checkpoint}"
    )

    checkpoint = torch.load(
        resume_checkpoint,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    completed_epoch_index = int(
        checkpoint.get("epoch", -1)
    )

    start_epoch = completed_epoch_index + 1

    best_val_loss = float(
        checkpoint.get(
            "best_val_loss",
            float("inf")
        )
    )

    best_epoch = int(
        checkpoint.get(
            "best_epoch",
            0
        )
    )

    no_improvement_epochs = int(
        checkpoint.get(
            "no_improvement_epochs",
            0
        )
    )

    saved_history = checkpoint.get(
        "history"
    )

    if saved_history is not None:
        history = saved_history

    print(
        f"Last completed epoch : "
        f"{completed_epoch_index + 1}"
    )

    print(
        f"Next epoch           : "
        f"{start_epoch + 1}"
    )

    print(
        f"Best epoch so far    : "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"No-improvement count : "
        f"{no_improvement_epochs}"
    )

    print("=" * 60)

else:

    print("\n" + "=" * 60)
    print("NO EPOCH-STUDY CHECKPOINT FOUND")
    print("Starting from epoch 1.")
    print("=" * 60)


# ======================================================
# Training Loop
# ======================================================

print("\n" + "=" * 60)
print("V3 EPOCH STUDY TRAINING STARTED")
print("=" * 60)
print("Preprocessing : Aspect-ratio preserving + padding")
print(f"CTC steps     : {CTC_TIME_STEPS}")
print("Validation    : 10%")
print("Best model    : Lowest validation CTC loss")
print(f"Maximum epochs: {MAX_EPOCHS}")
print(
    "Actual epoch count will be determined "
    "by validation-loss convergence."
)
print(
    f"Early stopping patience: "
    f"{EARLY_STOPPING_PATIENCE}"
)
print(
    f"Checkpoint directory: {CHECKPOINT_DIR}"
)
print("=" * 60 + "\n")


for epoch in range(
    start_epoch,
    MAX_EPOCHS
):

    print("\n" + "=" * 60)
    print(
        f"V3 Epoch {epoch + 1}/{MAX_EPOCHS}"
    )
    print("=" * 60)

    train_loss = train_one_epoch()

    val_loss = validate()

    scheduler.step(val_loss)

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print("\n" + "-" * 50)
    print(
        f"Epoch           : "
        f"{epoch + 1}/{MAX_EPOCHS}"
    )
    print(
        f"Training Loss   : "
        f"{train_loss:.4f}"
    )
    print(
        f"Validation Loss : "
        f"{val_loss:.4f}"
    )
    print(
        f"Learning Rate   : "
        f"{current_lr:.8f}"
    )
    print("-" * 50)

    history["epoch"].append(
        epoch + 1
    )

    history["train_loss"].append(
        float(train_loss)
    )

    history["val_loss"].append(
        float(val_loss)
    )

    history["learning_rate"].append(
        float(current_lr)
    )

    # --------------------------------------------------
    # Best model
    # --------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = float(val_loss)
        best_epoch = epoch + 1
        no_improvement_epochs = 0

        atomic_torch_save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("✓ BEST V3 MODEL UPDATED")
        print(
            f"Best Validation Loss: "
            f"{best_val_loss:.4f}"
        )
        print(
            f"Best Epoch: {best_epoch}"
        )

    else:

        no_improvement_epochs += 1

        print(
            "No validation improvement for "
            f"{no_improvement_epochs}/"
            f"{EARLY_STOPPING_PATIENCE} epoch(s)."
        )

    # --------------------------------------------------
    # Full checkpoint
    # --------------------------------------------------

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "no_improvement_epochs": int(
            no_improvement_epochs
        ),
        "ctc_time_steps": CTC_TIME_STEPS,
        "history": history,
        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "validation_ratio": VALIDATION_RATIO,
            "random_seed": RANDOM_SEED,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,
        }
    }

    # Rolling checkpoint for automatic resume.
    atomic_torch_save(
        checkpoint_data,
        LATEST_CHECKPOINT
    )

    # Permanent checkpoint for this exact epoch.
    epoch_checkpoint_path = (
        CHECKPOINT_DIR
        / f"checkpoint_epoch_{epoch + 1:03d}.pth"
    )

    atomic_torch_save(
        checkpoint_data,
        epoch_checkpoint_path
    )

    print(
        f"✓ Checkpoint saved: "
        f"{epoch_checkpoint_path}"
    )

    # --------------------------------------------------
    # Save history
    # --------------------------------------------------

    with open(
        HISTORY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "validation_loss",
            "learning_rate"
        ])

        for i in range(
            len(history["epoch"])
        ):

            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["val_loss"][i],
                history["learning_rate"][i]
            ])

    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------

    if (
        no_improvement_epochs
        >= EARLY_STOPPING_PATIENCE
    ):

        print("\n" + "=" * 60)
        print("EARLY STOPPING TRIGGERED")
        print(
            "Validation loss did not improve for "
            f"{EARLY_STOPPING_PATIENCE} "
            "consecutive epochs."
        )
        print("=" * 60)

        break


# ======================================================
# Final History Save
# ======================================================

with open(
    HISTORY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate"
    ])

    for i in range(
        len(history["epoch"])
    ):

        writer.writerow([
            history["epoch"][i],
            history["train_loss"][i],
            history["val_loss"][i],
            history["learning_rate"][i]
        ])


# ======================================================
# Plot
# ======================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

if history["val_loss"]:

    best_idx = min(
        range(len(history["val_loss"])),
        key=lambda i: history["val_loss"][i]
    )

    plot_best_epoch = (
        history["epoch"][best_idx]
    )

    plot_best_loss = (
        history["val_loss"][best_idx]
    )

    plt.axvline(
        plot_best_epoch,
        linestyle="--",
        label=(
            f"Best Epoch = "
            f"{plot_best_epoch}"
        )
    )

    plt.scatter(
        [plot_best_epoch],
        [plot_best_loss],
        s=80,
        label=(
            f"Best Val Loss = "
            f"{plot_best_loss:.4f}"
        )
    )

plt.xlabel("Epoch")
plt.ylabel("CTC Loss")
plt.title("V3 Training vs Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    HISTORY_PLOT,
    dpi=200
)

plt.show()


# ======================================================
# Epoch Analysis
# ======================================================

actual_epochs = len(
    history["epoch"]
)

if history["val_loss"]:

    best_idx = min(
        range(len(history["val_loss"])),
        key=lambda i: history["val_loss"][i]
    )

    best_epoch = history["epoch"][best_idx]
    best_val_loss = history["val_loss"][best_idx]

print("\n" + "=" * 60)
print("EPOCH ANALYSIS")
print("=" * 60)

print(
    f"Actual Epochs Trained   : "
    f"{actual_epochs}"
)

print(
    f"Best Epoch              : "
    f"{best_epoch}"
)

print(
    f"Best Validation Loss    : "
    f"{best_val_loss:.4f}"
)

print(
    f"Epoch history CSV       : "
    f"{HISTORY_CSV}"
)

print(
    f"Epoch analysis graph    : "
    f"{HISTORY_PLOT}"
)

if actual_epochs < MAX_EPOCHS:

    print(
        "Conclusion: training stopped "
        "automatically after "
        f"{actual_epochs} epochs."
    )

    print(
        f"Use epoch {best_epoch} as the "
        "selected training checkpoint "
        "because it achieved the minimum "
        "validation CTC loss."
    )

else:

    print(
        "Conclusion: the model reached "
        f"the {MAX_EPOCHS}-epoch safety ceiling."
    )

    print(
        "The required epoch count is not "
        "yet established. Increase the "
        "safety ceiling and repeat the "
        "epoch study if validation loss "
        "is still improving."
    )

print("=" * 60)


# ======================================================
# Final Model
# ======================================================

atomic_torch_save(
    model.state_dict(),
    FINAL_MODEL_PATH
)

print("\n" + "=" * 60)
print("V3 EPOCH STUDY COMPLETED")
print("=" * 60)

print(
    f"Latest checkpoint : "
    f"{LATEST_CHECKPOINT}"
)

print(
    f"Best model        : "
    f"{BEST_MODEL_PATH}"
)

print(
    f"Final model       : "
    f"{FINAL_MODEL_PATH}"
)

print(
    f"Actual epochs     : "
    f"{actual_epochs}"
)

print(
    f"Best epoch        : "
    f"{best_epoch}"
)

print(
    f"Best val loss     : "
    f"{best_val_loss:.4f}"
)

print(
    f"History CSV       : "
    f"{HISTORY_CSV}"
)

print(
    f"Loss graph        : "
    f"{HISTORY_PLOT}"
)

print(
    f"Checkpoints       : "
    f"{CHECKPOINT_DIR}"
)

print("=" * 60)
