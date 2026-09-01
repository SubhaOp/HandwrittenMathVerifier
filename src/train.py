import random
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv

from tqdm import tqdm

from torch.utils.data import (
    DataLoader,
    Subset
)

from torch.nn.utils.rnn import (
    pad_sequence
)

from src.config import *
from src.dataset import HMEDataset
from src.model import MathRecognizer


# ======================================================
# V3 Configuration
# ======================================================

VALIDATION_RATIO = 0.10

RANDOM_SEED = 42

# ======================================================
# EPOCH STUDY CONFIGURATION
# ======================================================
# 100 is ONLY a safety ceiling. It is NOT the required number of epochs.
# The actual training duration is determined automatically by validation-loss
# convergence and early stopping.
MAX_EPOCHS = 100
EPOCHS = MAX_EPOCHS  # safety ceiling only; early stopping determines actual duration

# IMPORTANT:
# Set this True for the FIRST epoch-study run so that the
# experiment starts from epoch 1 instead of resuming the old
# completed V3 run.
RESET_EPOCH_STUDY = False

# Early stopping:
# If validation loss does not improve for this many epochs,
# training stops automatically.
EARLY_STOPPING_PATIENCE = 8

# IMPORTANT:
# V3 now produces approximately 384 CTC time steps.
CTC_TIME_STEPS = 384


# ======================================================
# Reproducibility
# ======================================================

random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ======================================================
# Print Device
# ======================================================

print("\n========================================")
print("V3 TRAINING")
print("========================================")

print(
    f"Device : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("========================================\n")


# ======================================================
# Collate Function
# ======================================================

def collate_fn(batch):

    images = []

    labels = []

    for image, label in batch:

        images.append(image)

        labels.append(label)

    images = torch.stack(
        images
    )

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=0
    )

    return images, labels


# ======================================================
# Target Lengths
# ======================================================

def get_target_lengths(labels):

    """
    Padding / CTC blank ID = 0.

    Actual target characters use non-zero IDs.
    """

    return torch.tensor(

        [
            torch.count_nonzero(
                label
            ).item()

            for label in labels
        ],

        dtype=torch.long,

        device=labels.device
    )


# ======================================================
# Load Dataset
# ======================================================

print("\nLoading Dataset...")

dataset = HMEDataset()

print(
    f"Original Dataset Samples : "
    f"{len(dataset)}"
)


# ======================================================
# CTC Validity Filtering
# ======================================================

print(
    "\nChecking CTC validity..."
)

valid_indices = []

invalid_indices = []


for idx in tqdm(

    range(len(dataset)),

    desc="Checking labels"
):

    row = dataset.df.iloc[idx]

    label = dataset.encode_label(
        row["label"]
    )

    target_length = len(label)

    # --------------------------------------------------
    # Repeated adjacent labels require an extra CTC
    # blank position.
    # --------------------------------------------------

    if target_length > 1:

        repeats = int(

            (
                label[1:]
                ==
                label[:-1]
            )
            .sum()
            .item()
        )

    else:

        repeats = 0

    required_steps = (
        target_length
        +
        repeats
    )

    if required_steps <= CTC_TIME_STEPS:

        valid_indices.append(
            idx
        )

    else:

        invalid_indices.append(
            idx
        )


print("\n========================================")
print("V3 CTC Dataset Filtering")
print("========================================")

print(
    f"Original Samples : "
    f"{len(dataset)}"
)

print(
    f"Valid Samples    : "
    f"{len(valid_indices)}"
)

print(
    f"Filtered Samples : "
    f"{len(invalid_indices)}"
)

print("========================================")


# ======================================================
# Train / Validation Split
# ======================================================

generator = torch.Generator()

generator.manual_seed(
    RANDOM_SEED
)


permutation = torch.randperm(

    len(valid_indices),

    generator=generator
).tolist()


validation_size = int(

    len(valid_indices)
    *
    VALIDATION_RATIO
)

validation_size = max(
    1,
    validation_size
)


val_positions = permutation[
    :validation_size
]

train_positions = permutation[
    validation_size:
]


train_indices = [

    valid_indices[i]

    for i in train_positions
]


val_indices = [

    valid_indices[i]

    for i in val_positions
]


train_dataset = Subset(

    dataset,

    train_indices
)


val_dataset = Subset(

    dataset,

    val_indices
)


print("\n========================================")
print("V3 Dataset Split")
print("========================================")

print(
    f"Training Samples   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation Samples : "
    f"{len(val_dataset)}"
)

print("========================================")


# ======================================================
# DataLoaders
# ======================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),

    collate_fn=collate_fn
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),

    collate_fn=collate_fn
)


# ======================================================
# Create Model
# ======================================================

num_classes = (
    len(dataset.char2idx)
    + 1
)


model = MathRecognizer(
    num_classes
).to(DEVICE)


print("\nModel Created Successfully")

print(
    f"Number of Classes : "
    f"{num_classes}"
)

print(model)


# ======================================================
# Verify Sequence Length
# ======================================================

print(
    "\nChecking V3 model output shape..."
)


with torch.no_grad():

    test_input = torch.zeros(

        1,
        CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,

        device=DEVICE
    )

    test_output = model(
        test_input
    )


print(
    f"Input Shape  : "
    f"{tuple(test_input.shape)}"
)

print(
    f"Output Shape : "
    f"{tuple(test_output.shape)}"
)


if test_output.shape[1] != CTC_TIME_STEPS:

    raise RuntimeError(

        "V3 sequence length mismatch! "

        f"Expected {CTC_TIME_STEPS}, "

        f"got {test_output.shape[1]}"
    )


print(
    "✓ V3 sequence length verified."
)


del test_input
del test_output

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ======================================================
# CTC Loss
# ======================================================

criterion = nn.CTCLoss(

    blank=0,

    zero_infinity=True
)


# ======================================================
# Optimizer
# ======================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


# ======================================================
# Scheduler
# ======================================================

scheduler = (

    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=2,

        min_lr=1e-6
    )
)


# ======================================================
# V3 Model Paths
#
# IMPORTANT:
# These are completely separate from V2.
# ======================================================

checkpoint_path = (

    MODEL_DIR
    /
    "checkpoint_v3.pth"
)


best_model_path = (

    MODEL_DIR
    /
    "best_model_v3.pth"
)


final_model_path = (

    MODEL_DIR
    /
    "math_recognizer_v3.pth"
)


# ======================================================
# Epoch-study output files, checkpoints and history
# ======================================================
history_csv_path = MODEL_DIR / "v3_epoch_history.csv"
history_plot_path = MODEL_DIR / "v3_epoch_analysis.png"

# Keep the epoch experiment separate from the normal V3 model files.
checkpoint_path = MODEL_DIR / "checkpoint_v3_epoch_study.pth"
best_model_path = MODEL_DIR / "best_model_v3_epoch_study.pth"
final_model_path = MODEL_DIR / "math_recognizer_v3_epoch_study.pth"

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "learning_rate": [],
}


# ======================================================
# Train One Epoch
# ======================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0


    progress_bar = tqdm(

        train_loader,

        desc="Training"
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


        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        outputs = model(
            images
        )


        outputs = outputs.log_softmax(
            dim=2
        )


        # B,T,C -> T,B,C

        outputs = outputs.permute(
            1,
            0,
            2
        )


        # --------------------------------------------------
        # Input lengths
        # --------------------------------------------------

        input_lengths = torch.full(

            size=(
                images.size(0),
            ),

            fill_value=outputs.size(0),

            dtype=torch.long,

            device=DEVICE
        )


        # --------------------------------------------------
        # Target lengths
        # --------------------------------------------------

        target_lengths = (
            get_target_lengths(
                labels
            )
        )


        # --------------------------------------------------
        # CTC Loss
        # --------------------------------------------------

        loss = criterion(

            outputs,

            labels,

            input_lengths,

            target_lengths
        )


        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        loss.backward()


        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0
        )


        optimizer.step()


        total_loss += (
            loss.item()
        )


        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}",

            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )


    average_loss = (

        total_loss
        /
        len(train_loader)
    )


    return average_loss


# ======================================================
# Validation
# ======================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0


    progress_bar = tqdm(

        val_loader,

        desc="Validation"
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


        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        outputs = model(
            images
        )


        outputs = outputs.log_softmax(
            dim=2
        )


        outputs = outputs.permute(
            1,
            0,
            2
        )


        # --------------------------------------------------
        # Input lengths
        # --------------------------------------------------

        input_lengths = torch.full(

            size=(
                images.size(0),
            ),

            fill_value=outputs.size(0),

            dtype=torch.long,

            device=DEVICE
        )


        # --------------------------------------------------
        # Target lengths
        # --------------------------------------------------

        target_lengths = (
            get_target_lengths(
                labels
            )
        )


        # --------------------------------------------------
        # Validation CTC loss
        # --------------------------------------------------

        loss = criterion(

            outputs,

            labels,

            input_lengths,

            target_lengths
        )


        total_loss += (
            loss.item()
        )


        progress_bar.set_postfix(

            val_loss=
            f"{loss.item():.4f}"
        )


    average_loss = (

        total_loss
        /
        len(val_loader)
    )


    return average_loss


# ======================================================
# Epoch-study start / resume logic
# ======================================================
# RESET_EPOCH_STUDY = False: resume from the latest checkpoint.
# If latest_checkpoint.pth is missing, the highest numbered
# checkpoint_epoch_XXX.pth is used automatically.

start_epoch = 0
best_val_loss = float("inf")
best_epoch = 0
no_improvement_epochs = 0
resume_checkpoint = None

if RESET_EPOCH_STUDY:
    print("\nEpoch-study mode: STARTING FROM SCRATCH")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
else:
    if checkpoint_path.exists():
        print("\nEpoch-study checkpoint found. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", history)
        print(f"Resume epoch: {start_epoch + 1}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print("\nNo epoch-study checkpoint found. Starting from scratch.")


# ======================================================
# Training
# ======================================================

print("\n========================================")
print("V3 EPOCH STUDY TRAINING STARTED")
print("========================================")

print(
    "Preprocessing : "
    "Aspect-ratio preserving + padding"
)

print(
    "CTC steps     : "
    "384"
)

print(
    "Validation    : "
    "10%"
)

print(
    "Best model    : "
    "Lowest validation CTC loss"
)

print(
    f"Maximum epoch safety ceiling: {MAX_EPOCHS}"
)

print(
    "Actual epoch count will be determined automatically "
    "from validation-loss convergence."
)

print(
    f"Early stopping patience: {EARLY_STOPPING_PATIENCE}"
)

print(
    "Model files   : "
    "V3 only"
)

print("========================================\n")


for epoch in range(

    start_epoch,

    MAX_EPOCHS
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"V3 Epoch "
        f"{epoch + 1}/{MAX_EPOCHS}"
    )

    print(
        f"{'=' * 60}"
    )


    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    train_loss = (
        train_one_epoch()
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    val_loss = (
        validate()
    )


    # --------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------

    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    print("\n----------------------------------------")

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

    print("----------------------------------------")

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["learning_rate"].append(float(current_lr))

    # ==================================================
    # Update best epoch / early stopping state
    # ==================================================

    if val_loss < best_val_loss:
        best_val_loss = float(val_loss)
        best_epoch = epoch + 1
        no_improvement_epochs = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("✓ BEST V3 MODEL UPDATED")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Best Epoch: {best_epoch}")
    else:
        no_improvement_epochs += 1
        print(
            f"No validation improvement for "
            f"{no_improvement_epochs}/{EARLY_STOPPING_PATIENCE} epoch(s)."
        )

    # ==================================================
    # Save checkpoint after EVERY epoch
    # ==================================================

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "no_improvement_epochs": int(no_improvement_epochs),
        "ctc_time_steps": CTC_TIME_STEPS,
        "history": history,
    }

    # One rolling checkpoint used for automatic resume.
    torch.save(
        checkpoint_data,
        checkpoint_path
    )

    # A permanent checkpoint for this specific epoch.
    epoch_checkpoint_path = (
        CHECKPOINT_DIR / f"checkpoint_epoch_{epoch + 1:03d}.pth"
    )
    torch.save(
        checkpoint_data,
        epoch_checkpoint_path
    )

    print(f"✓ Checkpoint saved: {epoch_checkpoint_path}")

    # Save the history after every epoch as well.
    with open(history_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "validation_loss",
            "learning_rate"
        ])

        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["val_loss"][i],
                history["learning_rate"][i],
            ])

    if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
        print("\nEARLY STOPPING TRIGGERED")
        print(
            f"Validation loss did not improve for "
            f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
        )
        break


# ======================================================
# Save epoch history
# ======================================================

with open(history_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate"
    ])

    for i in range(len(history["epoch"])):
        writer.writerow([
            history["epoch"][i],
            history["train_loss"][i],
            history["val_loss"][i],
            history["learning_rate"][i],
        ])


# ======================================================
# Plot training/validation loss
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
    best_epoch = history["epoch"][best_idx]
    best_loss = history["val_loss"][best_idx]

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best Epoch = {best_epoch}"
    )

    plt.scatter(
        [best_epoch],
        [best_loss],
        s=80,
        label=f"Best Val Loss = {best_loss:.4f}"
    )

plt.xlabel("Epoch")
plt.ylabel("CTC Loss")
plt.title("V3 Training vs Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(history_plot_path, dpi=200)
plt.show()

print("\n========================================")
print("EPOCH ANALYSIS")
print("========================================")
actual_epochs = len(history["epoch"])

print(f"Actual Epochs Trained   : {actual_epochs}")
print(f"Best Epoch              : {best_epoch}")
print(f"Best Validation Loss    : {best_val_loss:.4f}")
print(f"Epoch history CSV       : {history_csv_path}")
print(f"Epoch analysis graph    : {history_plot_path}")

if actual_epochs < MAX_EPOCHS:
    print(
        "Conclusion: training stopped automatically after "
        f"{actual_epochs} epochs."
    )
    print(
        f"Use epoch {best_epoch} as the selected training checkpoint "
        "because it achieved the minimum validation CTC loss."
    )
else:
    print(
        "Conclusion: the model was still improving enough to reach "
        f"the safety ceiling of {MAX_EPOCHS} epochs."
    )
    print(
        "The required epoch count is not yet established; "
        "increase the safety ceiling and repeat the experiment."
    )

print("========================================")


# ======================================================
# Save Final Model
# ======================================================

torch.save(

    model.state_dict(),

    final_model_path
)


print("\n========================================")
print("V3 TRAINING COMPLETED")
print("========================================")

print(
    f"Checkpoint : "
    f"{checkpoint_path}"
)

print(
    f"Best Model : "
    f"{best_model_path}"
)

print(
    f"Final Model: "
    f"{final_model_path}"
)

print(f"Actual Epochs Trained: {actual_epochs}")
print(f"Best Epoch (minimum validation loss): {best_epoch}")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Epoch history CSV: {history_csv_path}")
print(f"Epoch analysis graph: {history_plot_path}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print(f"Best model: {best_model_path}")

if actual_epochs < MAX_EPOCHS:
    print(f"Conclusion: training stopped automatically after {actual_epochs} epochs.")
    print(f"Selected epoch for the report: {best_epoch}")
else:
    print(f"Conclusion: the safety ceiling of {MAX_EPOCHS} epochs was reached.")
    print("Increase MAX_EPOCHS and repeat the epoch study before claiming convergence.")

print("========================================")