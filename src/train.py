"""
V3 EPOCH STUDY -- CLEANED DATASET
====================================

Runs the same V3 architecture/training regime as before, but against
the cleaned dataset (once config.py resolves to HME100K_CLEANED and
the vocab has been rebuilt from cleaned train_labels.txt -- see the
2 steps in chat before running this).

Determines how many epochs the CLEANED dataset actually needs, the
same way the original epoch study did: 90/10 train/val split, early
stopping on validation CTC loss, CSV history + loss-curve plot so you
can see the answer instead of guessing it.

Checkpoints use a "_cleaned" suffix so this can never be confused with,
or accidentally resumed from, a checkpoint trained on the original
uncleaned data (those have the same 245-class output shape, so loading
one into the other would NOT crash -- it would just silently mix
training provenance, which is worse than a crash because nothing
would tell you it happened).

Outputs:
    {OUTPUT_DIR}/v3_cleaned_epoch_history.csv
    {OUTPUT_DIR}/v3_cleaned_epoch_analysis.png
    {MODEL_DIR}/checkpoint_v3_cleaned.pth      (latest, for resume)
    {MODEL_DIR}/best_model_v3_cleaned.pth      (lowest val loss -- use this one)
"""

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
# Configuration
# ======================================================

VALIDATION_RATIO = 0.10
RANDOM_SEED = 42
CTC_TIME_STEPS = 384

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 8

# Set True only if you deliberately want to discard the checkpoints
# below and start this cleaned-data study over from epoch 1.
RESET_EPOCH_STUDY = False

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ======================================================
# Sanity check: confirm the CLEANED data is actually what
# gets loaded, before spending hours training on the wrong
# thing.
#
# CHANGED: checks the exact sample count (74,226 -- the
# confirmed count from your post_cleaning_audit_summary.json)
# instead of checking for "CLEANED" in the path string. That
# string check only worked if the cleaned dataset lived in a
# separate HME100K_CLEANED folder; if you instead replaced the
# original HME100K folder's contents in place, DATASET_DIR
# still just says ".../HME100K" even though the data itself is
# the cleaned version -- the old check would have wrongly
# blocked you. A count check is correct either way.
# ======================================================

EXPECTED_CLEAN_TRAIN_COUNT = 74226

print("\n" + "=" * 60)
print("V3 EPOCH STUDY -- CLEANED DATASET")
print("=" * 60)
print(f"Device      : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"Dataset dir : {DATASET_DIR}")
print("=" * 60 + "\n")


# ======================================================
# Collate function
# ======================================================

def collate_fn(batch):
    images, labels = [], []
    for image, label in batch:
        images.append(image)
        labels.append(label)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return images, labels


def get_target_lengths(labels):
    """Padding / CTC blank ID = 0. Actual target symbols use non-zero IDs."""
    return torch.tensor(
        [torch.count_nonzero(label).item() for label in labels],
        dtype=torch.long,
        device=labels.device,
    )


# ======================================================
# Load dataset (now resolves to the cleaned one)
# ======================================================

print("Loading dataset...")
dataset = HMEDataset()
print(f"Dataset samples loaded : {len(dataset)}")

if len(dataset) != EXPECTED_CLEAN_TRAIN_COUNT:
    raise RuntimeError(
        f"\nExpected exactly {EXPECTED_CLEAN_TRAIN_COUNT} training samples "
        f"(the confirmed count from your post-cleaning audit), but "
        f"HMEDataset() loaded {len(dataset)}.\n\n"
        f"Dataset dir currently resolves to: {DATASET_DIR}\n\n"
        f"This means the original (74,502-row) dataset is probably "
        f"still what's being loaded, not the cleaned one -- fix this "
        f"before training, since hours of training on the wrong data "
        f"can't be undone after the fact.\n\n"
        f"If you're confident {len(dataset)} is correct for a reason "
        f"not listed above (e.g. you cleaned further since the audit "
        f"summary was generated), update EXPECTED_CLEAN_TRAIN_COUNT at "
        f"the top of this script to match."
    )

print(f"Confirmed: sample count matches the cleaned dataset ({EXPECTED_CLEAN_TRAIN_COUNT}).")


# ======================================================
# CTC validity filtering (unchanged logic -- verified
# against your existing train script)
# ======================================================

print("\nChecking CTC validity...")

valid_indices, invalid_indices = [], []

for idx in tqdm(range(len(dataset)), desc="Checking labels"):
    row = dataset.df.iloc[idx]
    label = dataset.encode_label(row["label"])
    target_length = len(label)

    if target_length > 1:
        repeats = int((label[1:] == label[:-1]).sum().item())
    else:
        repeats = 0

    required_steps = target_length + repeats

    if required_steps <= CTC_TIME_STEPS:
        valid_indices.append(idx)
    else:
        invalid_indices.append(idx)

print("\n" + "=" * 60)
print("CTC Dataset Filtering (cleaned data)")
print("=" * 60)
print(f"Original Samples : {len(dataset)}")
print(f"Valid Samples    : {len(valid_indices)}")
print(f"Filtered Samples : {len(invalid_indices)}")
print("=" * 60)


# ======================================================
# Train / validation split
# ======================================================

generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)

permutation = torch.randperm(len(valid_indices), generator=generator).tolist()
validation_size = max(1, int(len(valid_indices) * VALIDATION_RATIO))

val_positions = permutation[:validation_size]
train_positions = permutation[validation_size:]

train_indices = [valid_indices[i] for i in train_positions]
val_indices = [valid_indices[i] for i in val_positions]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

print("\n" + "=" * 60)
print("Dataset Split")
print("=" * 60)
print(f"Training Samples   : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")
print("=" * 60)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)


# ======================================================
# Model
# ======================================================

num_classes = len(dataset.char2idx) + 1
model = MathRecognizer(num_classes).to(DEVICE)

print(f"\nModel created. Number of classes: {num_classes}")

with torch.no_grad():
    test_input = torch.zeros(1, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, device=DEVICE)
    test_output = model(test_input)

if test_output.shape[1] != CTC_TIME_STEPS:
    raise RuntimeError(
        f"Sequence length mismatch! Expected {CTC_TIME_STEPS}, got {test_output.shape[1]}"
    )

print(f"Output shape verified: {tuple(test_output.shape)}")
del test_input, test_output
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================
# Loss / optimizer / scheduler
# ======================================================

criterion = nn.CTCLoss(blank=0, zero_infinity=True)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
)


# ======================================================
# Paths (distinct "_cleaned" names -- see module docstring)
# ======================================================

checkpoint_path = MODEL_DIR / "checkpoint_v3_cleaned.pth"
best_model_path = MODEL_DIR / "best_model_v3_cleaned.pth"

HISTORY_CSV = OUTPUT_DIR / "v3_cleaned_epoch_history.csv"
HISTORY_PNG = OUTPUT_DIR / "v3_cleaned_epoch_analysis.png"


# ======================================================
# Train / validate one epoch
# ======================================================

def train_one_epoch():
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training")

    for images, labels in progress_bar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        outputs = outputs.log_softmax(dim=2)
        outputs = outputs.permute(1, 0, 2)  # B,T,C -> T,B,C

        input_lengths = torch.full(
            size=(images.size(0),), fill_value=outputs.size(0),
            dtype=torch.long, device=DEVICE,
        )
        target_lengths = get_target_lengths(labels)

        loss = criterion(outputs, labels, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    return total_loss / len(train_loader)


@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(val_loader, desc="Validation")

    for images, labels in progress_bar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        outputs = outputs.log_softmax(dim=2)
        outputs = outputs.permute(1, 0, 2)

        input_lengths = torch.full(
            size=(images.size(0),), fill_value=outputs.size(0),
            dtype=torch.long, device=DEVICE,
        )
        target_lengths = get_target_lengths(labels)

        loss = criterion(outputs, labels, input_lengths, target_lengths)
        total_loss += loss.item()
        progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / len(val_loader)


# ======================================================
# History logging + plotting
# ======================================================

def log_epoch(epoch, train_loss, val_loss, lr):
    write_header = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"])
        if write_header:
            writer.writeheader()
        writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr})


def plot_history():
    epochs, train_losses, val_losses = [], [], []
    with open(HISTORY_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o", markersize=3)
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", markersize=3)
    plt.axvline(epochs[best_idx], color="gray", linestyle="--", alpha=0.6)
    plt.scatter([epochs[best_idx]], [val_losses[best_idx]], color="red", zorder=5,
                label=f"Best epoch {epochs[best_idx]} (val_loss={val_losses[best_idx]:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.title("V3 Training vs Validation Loss -- Cleaned Dataset")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(HISTORY_PNG, dpi=150)
    plt.close()

    return epochs[best_idx], val_losses[best_idx]


# ======================================================
# Resume (with safe fallback if incompatible)
# ======================================================

start_epoch = 0
best_val_loss = float("inf")
best_epoch = 0
no_improvement_epochs = 0

if RESET_EPOCH_STUDY:
    print("\nRESET_EPOCH_STUDY=True -- ignoring any existing checkpoint.")

elif checkpoint_path.exists():
    print(f"\nCheckpoint found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_epoch = checkpoint.get("best_epoch", 0)
        no_improvement_epochs = checkpoint.get("no_improvement_epochs", 0)

        print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Best val loss so far: {best_val_loss:.4f} (epoch {best_epoch})")

    except RuntimeError as e:
        print("\n" + "=" * 60)
        print("CHECKPOINT INCOMPATIBLE -- STARTING FRESH")
        print("=" * 60)
        print(f"Error: {e}")
        print(f"The old checkpoint at {checkpoint_path} was NOT deleted.")
        start_epoch = 0
        best_val_loss = float("inf")
        best_epoch = 0
        no_improvement_epochs = 0

else:
    print("\nNo checkpoint found -- starting fresh.")


# ======================================================
# Main loop
# ======================================================

if start_epoch >= MAX_EPOCHS:
    print(f"\nAlready completed {start_epoch}/{MAX_EPOCHS} epochs. Nothing to do.")
else:
    for epoch in range(start_epoch, MAX_EPOCHS):
        print(f"\n{'=' * 60}\nEPOCH {epoch + 1}/{MAX_EPOCHS}\n{'=' * 60}")

        train_loss = train_one_epoch()
        val_loss = validate()
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n{'-' * 60}")
        print(f"Epoch           : {epoch + 1}/{MAX_EPOCHS}")
        print(f"Training Loss   : {train_loss:.4f}")
        print(f"Validation Loss : {val_loss:.4f}")
        print(f"Learning Rate   : {current_lr:.8f}")
        print("-" * 60)

        log_epoch(epoch + 1, train_loss, val_loss, current_lr)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improvement_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (val_loss={val_loss:.4f})")
        else:
            no_improvement_epochs += 1
            print(f"No improvement for {no_improvement_epochs}/{EARLY_STOPPING_PATIENCE} epoch(s).")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "no_improvement_epochs": no_improvement_epochs,
        }, checkpoint_path)

        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"\n{'=' * 60}\nEARLY STOPPING TRIGGERED")
            print(f"Validation loss did not improve for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
            print("=" * 60)
            break

    actual_epochs = epoch + 1
    plotted_best_epoch, plotted_best_loss = plot_history()

    print(f"\n{'=' * 60}\nEPOCH ANALYSIS -- CLEANED DATASET\n{'=' * 60}")
    print(f"Actual Epochs Trained : {actual_epochs}")
    print(f"Best Epoch            : {best_epoch}")
    print(f"Best Validation Loss  : {best_val_loss:.4f}")
    print(f"Epoch history CSV     : {HISTORY_CSV}")
    print(f"Epoch analysis graph  : {HISTORY_PNG}")
    print(f"Best model checkpoint : {best_model_path}")
    print(f"\nUse epoch {best_epoch}'s checkpoint ({best_model_path.name}) -- ")
    print("that's the lowest validation loss, not the final epoch trained.")
    print("=" * 60)
