"""
FINAL V3 TRAINING — 16 EPOCHS, FULL VALID HME100K TRAINING SET

Architecture:
    HME100K
      -> existing V3 preprocessing
      -> ResNet18
      -> 384-step sequence
      -> 2-layer BiLSTM
      -> Linear
      -> CTC

IMPORTANT:
- Uses the existing 245-class vocabulary (244 tokens + CTC blank).
- Uses ALL CTC-valid HME100K training samples.
- Does NOT create a validation split.
- Trains for EXACTLY 16 completed epochs.
- Keeps V3 architecture, preprocessing, batch size, AdamW, LR,
  weight decay, CTC loss, and gradient clipping unchanged.
- ReduceLROnPlateau is not stepped because there is no validation set
  in this final full-data experiment; LR therefore remains exactly 1e-4.
- Saves checkpoints and the final model to Google Drive.
- Can resume after a Colab runtime interruption.
"""

import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.config import *
from src.dataset import HMEDataset
from src.model import MathRecognizer


# ======================================================
# FINAL TRAINING CONFIGURATION
# ======================================================

FINAL_EPOCHS = 16
RANDOM_SEED = 42
CTC_TIME_STEPS = 384

# Keep these identical to the V3 epoch-study configuration.
BATCH_SIZE_FINAL = BATCH_SIZE          # 32
LEARNING_RATE_FINAL = LEARNING_RATE    # 1e-4
WEIGHT_DECAY_FINAL = 1e-4
GRAD_CLIP_MAX_NORM = 5.0

# False = resume if a final-training checkpoint exists.
# True  = intentionally start this final experiment from scratch.
RESET_FINAL_TRAINING = False


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
# Google Drive storage
# ======================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/HandwrittenMathVerifier")

if DRIVE_ROOT.exists():
    FINAL_DIR = DRIVE_ROOT / "final_training_16_epoch"
    print("\nStorage mode : GOOGLE DRIVE")
else:
    FINAL_DIR = MODEL_DIR / "final_training_16_epoch"
    print("\nStorage mode : LOCAL/FALLBACK")

CHECKPOINT_DIR = FINAL_DIR / "checkpoints"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pth"
FINAL_MODEL_PATH = FINAL_DIR / "math_recognizer_v3_final_epoch16.pth"
FINAL_HISTORY_CSV = FINAL_DIR / "v3_final_training_history.csv"


# ======================================================
# Device
# ======================================================

print("\n" + "=" * 70)
print("V3 FINAL TRAINING — 16 EPOCHS / FULL VALID TRAINING SET")
print("=" * 70)
print("Device :", DEVICE)

if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
    print(
        "VRAM   :",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

print("=" * 70)


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
    # 0 is CTC blank and also the padding value.
    # All real vocabulary tokens have non-zero IDs.
    return torch.tensor(
        [
            torch.count_nonzero(label).item()
            for label in labels
        ],
        dtype=torch.long,
        device=labels.device
    )


# ======================================================
# Atomic checkpoint save
# ======================================================

def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp_path)
    temp_path.replace(path)


# ======================================================
# Load full training dataset
# ======================================================

print("\nLoading HME100K training dataset...")

dataset = HMEDataset()

print(f"Original training samples : {len(dataset)}")


# ======================================================
# CTC validity filtering
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
            (label[1:] == label[:-1]).sum().item()
        )
    else:
        repeats = 0

    required_steps = target_length + repeats

    if required_steps <= CTC_TIME_STEPS:
        valid_indices.append(idx)
    else:
        invalid_indices.append(idx)

print("\n" + "=" * 70)
print("FINAL TRAINING DATASET")
print("=" * 70)
print(f"Original samples : {len(dataset)}")
print(f"Valid samples    : {len(valid_indices)}")
print(f"Filtered samples : {len(invalid_indices)}")
print("Validation split : NONE")
print("=" * 70)

if len(valid_indices) == 0:
    raise RuntimeError("No valid training samples were found.")


# ======================================================
# Full-data loader
# ======================================================

# IMPORTANT:
# Unlike the epoch study, there is NO 90/10 train-validation split.
# Every CTC-valid sample is used for training.

train_dataset = torch.utils.data.Subset(
    dataset,
    valid_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_FINAL,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
    persistent_workers=(NUM_WORKERS > 0)
)

print(f"\nTraining samples actually used : {len(train_dataset)}")
print(f"Batch size                     : {BATCH_SIZE_FINAL}")
print(f"Batches per epoch              : {len(train_loader)}")


# ======================================================
# Model
# ======================================================

num_classes = len(dataset.char2idx) + 1

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)
print(f"Vocabulary tokens : {len(dataset.char2idx)}")
print(f"Output classes    : {num_classes}")
print("Expected          : 245 (244 vocabulary tokens + CTC blank)")
print("=" * 70)

if num_classes != 245:
    raise RuntimeError(
        f"Expected 245 output classes for the corrected vocabulary, "
        f"but found {num_classes}. Check char2idx.json."
    )

model = MathRecognizer(num_classes).to(DEVICE)


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

print(f"Input shape  : {tuple(test_input.shape)}")
print(f"Output shape : {tuple(test_output.shape)}")

if test_output.shape[1] != CTC_TIME_STEPS:
    raise RuntimeError(
        f"V3 sequence length mismatch: expected {CTC_TIME_STEPS}, "
        f"got {test_output.shape[1]}"
    )

print("V3 sequence length verified: 384")

del test_input
del test_output

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================
# Loss / Optimizer
# ======================================================

criterion = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE_FINAL,
    weight_decay=WEIGHT_DECAY_FINAL
)


# ======================================================
# Training history
# ======================================================

history = {
    "epoch": [],
    "train_loss": [],
    "learning_rate": []
}


# ======================================================
# Train one epoch
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

        optimizer.zero_grad(set_to_none=True)

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
            max_norm=GRAD_CLIP_MAX_NORM
        )

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    return total_loss / len(train_loader)


# ======================================================
# Resume logic
# ======================================================

start_epoch = 0

if RESET_FINAL_TRAINING:
    print("\nStarting FINAL 16-EPOCH training from scratch.")

elif LATEST_CHECKPOINT.exists():

    print("\n" + "=" * 70)
    print("RESUMING FINAL TRAINING")
    print("=" * 70)
    print(f"Checkpoint : {LATEST_CHECKPOINT}")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    completed_epoch_index = int(
        checkpoint.get("epoch", -1)
    )

    start_epoch = completed_epoch_index + 1

    saved_history = checkpoint.get("history")

    if saved_history is not None:
        history = saved_history

    print(
        f"Last completed epoch : {completed_epoch_index + 1}"
    )
    print(
        f"Next epoch           : {start_epoch + 1}"
    )
    print("=" * 70)

else:
    print("\nNo final-training checkpoint found.")
    print("Starting from epoch 1.")


if start_epoch >= FINAL_EPOCHS:
    print("\nAll 16 epochs are already completed.")
else:

    # ==================================================
    # EXACTLY 16 EPOCHS
    # ==================================================

    print("\n" + "=" * 70)
    print("FINAL TRAINING STARTED")
    print("=" * 70)
    print("Architecture       : V3 ResNet18 + 2-layer BiLSTM + Linear + CTC")
    print("Vocabulary          : 245 classes including CTC blank")
    print("CTC time steps      : 384")
    print("Training samples    :", len(train_dataset))
    print("Validation split    : NONE")
    print("Batch size          :", BATCH_SIZE_FINAL)
    print("Learning rate       :", LEARNING_RATE_FINAL)
    print("Optimizer           : AdamW")
    print("Weight decay        :", WEIGHT_DECAY_FINAL)
    print("Gradient clip       :", GRAD_CLIP_MAX_NORM)
    print("Total epochs        :", FINAL_EPOCHS)
    print("LR schedule         : Fixed 1e-4 (no validation available)")
    print("Checkpoint directory:", CHECKPOINT_DIR)
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, FINAL_EPOCHS):

        print("\n" + "=" * 70)
        print(f"FINAL V3 EPOCH {epoch + 1}/{FINAL_EPOCHS}")
        print("=" * 70)

        train_loss = train_one_epoch()

        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "-" * 60)
        print(f"Epoch         : {epoch + 1}/{FINAL_EPOCHS}")
        print(f"Training Loss : {train_loss:.6f}")
        print(f"Learning Rate : {current_lr:.8f}")
        print("-" * 60)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["learning_rate"].append(float(current_lr))

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(train_loss),
            "ctc_time_steps": CTC_TIME_STEPS,
            "num_classes": num_classes,
            "vocabulary_size": len(dataset.char2idx),
            "training_samples": len(train_dataset),
            "history": history,
            "config": {
                "epochs": FINAL_EPOCHS,
                "batch_size": BATCH_SIZE_FINAL,
                "learning_rate": LEARNING_RATE_FINAL,
                "weight_decay": WEIGHT_DECAY_FINAL,
                "gradient_clip": GRAD_CLIP_MAX_NORM,
                "random_seed": RANDOM_SEED,
                "validation_ratio": 0.0,
            }
        }

        # Rolling checkpoint for Colab interruption recovery.
        atomic_torch_save(
            checkpoint_data,
            LATEST_CHECKPOINT
        )

        # Permanent checkpoint for this exact epoch.
        epoch_checkpoint_path = (
            CHECKPOINT_DIR /
            f"checkpoint_epoch_{epoch + 1:03d}.pth"
        )

        atomic_torch_save(
            checkpoint_data,
            epoch_checkpoint_path
        )

        # Save CSV after every epoch.
        with open(
            FINAL_HISTORY_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "learning_rate"
            ])

            for i in range(len(history["epoch"])):
                writer.writerow([
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["learning_rate"][i]
                ])

        print(
            f"✓ Checkpoint saved: {epoch_checkpoint_path}"
        )


# ======================================================
# Save final model
# ======================================================

atomic_torch_save(
    model.state_dict(),
    FINAL_MODEL_PATH
)

print("\n" + "=" * 70)
print("FINAL V3 TRAINING COMPLETED")
print("=" * 70)
print(f"Epochs completed : {len(history['epoch'])}")
print(f"Final model      : {FINAL_MODEL_PATH}")
print(f"Latest checkpoint: {LATEST_CHECKPOINT}")
print(f"History CSV      : {FINAL_HISTORY_CSV}")
print(f"Checkpoint dir   : {CHECKPOINT_DIR}")

if history["train_loss"]:
    print(
        f"Epoch 1 loss     : {history['train_loss'][0]:.6f}"
    )
    print(
        f"Epoch 16 loss    : {history['train_loss'][-1]:.6f}"
    )

print("=" * 70)

print("\nIMPORTANT:")
print("Now evaluate ONLY this final model on the independent HME100K test set.")
print("Do not use the training/epoch-study validation set as the test set.")
