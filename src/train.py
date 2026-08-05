import random

import torch
import torch.nn as nn

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
# Configuration
# ======================================================

VALIDATION_RATIO = 0.10

RANDOM_SEED = 42

CTC_TIME_STEPS = 192


# ======================================================
# Reproducibility
# ======================================================

random.seed(RANDOM_SEED)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


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
# CTC Target Length
# ======================================================

def get_target_lengths(labels):

    """
    Vocabulary IDs start from 1.
    CTC blank/padding ID is 0.

    Therefore non-zero values represent
    actual target characters.
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
# Dataset
# ======================================================

print("\nLoading Dataset...")

dataset = HMEDataset()

print(
    f"Original Dataset Samples : "
    f"{len(dataset)}"
)


# ======================================================
# Find CTC-valid samples
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

    # Do not load image during this scan.

    row = dataset.df.iloc[idx]

    label = dataset.encode_label(
        row["label"]
    )

    target_length = len(label)

    # ----------------------------------------------
    # CTC repeated-character requirement
    # ----------------------------------------------

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
        target_length +
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
print("CTC Dataset Filtering")
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
print("Dataset Split")
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
# Model
# ======================================================

num_classes = (
    len(dataset.char2idx)
    + 1
)


model = MathRecognizer(
    num_classes
).to(DEVICE)


print("\nModel Created Successfully")

print(model)


# ======================================================
# Loss
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
# Learning Rate Scheduler
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
# Model Paths
#
# IMPORTANT:
# New names are intentional.
# Your original baseline is NOT overwritten.
# ======================================================

checkpoint_path = (
    MODEL_DIR /
    "checkpoint_v2.pth"
)

best_model_path = (
    MODEL_DIR /
    "best_model_v2.pth"
)

final_model_path = (
    MODEL_DIR /
    "math_recognizer_v2.pth"
)


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


        # ----------------------------------------------
        # Forward
        # ----------------------------------------------

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


        # ----------------------------------------------
        # Sequence lengths
        # ----------------------------------------------

        input_lengths = torch.full(

            size=(
                images.size(0),
            ),

            fill_value=outputs.size(0),

            dtype=torch.long,

            device=DEVICE
        )


        target_lengths = (
            get_target_lengths(
                labels
            )
        )


        # ----------------------------------------------
        # CTC Loss
        # ----------------------------------------------

        loss = criterion(

            outputs,

            labels,

            input_lengths,

            target_lengths
        )


        # ----------------------------------------------
        # Backpropagation
        # ----------------------------------------------

        loss.backward()


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

        total_loss /
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


        # ----------------------------------------------
        # Forward
        # ----------------------------------------------

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


        # ----------------------------------------------
        # Lengths
        # ----------------------------------------------

        input_lengths = torch.full(

            size=(
                images.size(0),
            ),

            fill_value=outputs.size(0),

            dtype=torch.long,

            device=DEVICE
        )


        target_lengths = (
            get_target_lengths(
                labels
            )
        )


        # ----------------------------------------------
        # Validation Loss
        # ----------------------------------------------

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

        total_loss /
        len(val_loader)
    )


    return average_loss


# ======================================================
# Resume Training
# ======================================================

start_epoch = 0

best_val_loss = float(
    "inf"
)


if checkpoint_path.exists():

    print(
        "\nLoading V2 Checkpoint..."
    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    optimizer.load_state_dict(

        checkpoint[
            "optimizer_state_dict"
        ]
    )


    if "scheduler_state_dict" in checkpoint:

        scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]
        )


    start_epoch = (

        checkpoint["epoch"]
        + 1
    )


    best_val_loss = checkpoint.get(

        "best_val_loss",

        float("inf")
    )


    print(
        f"Resuming from Epoch "
        f"{start_epoch + 1}"
    )


    print(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}"
    )


else:

    print(
        "\nStarting Fresh V2 Training..."
    )


# ======================================================
# Training
# ======================================================

print("\n========================================")
print("V2 TRAINING STARTED")
print("========================================")

print(
    "Preprocessing : "
    "Aspect-ratio preserving resize"
)

print(
    "Validation    : "
    "10%"
)

print(
    "Best model by : "
    "Validation CTC loss"
)

print("========================================\n")


for epoch in range(
    start_epoch,
    EPOCHS
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        f"{'=' * 60}"
    )


    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    train_loss = (
        train_one_epoch()
    )


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    val_loss = (
        validate()
    )


    # ----------------------------------------------
    # Scheduler
    # ----------------------------------------------

    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    print("\n----------------------------------------")

    print(
        f"Epoch            : "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss    : "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation Loss  : "
        f"{val_loss:.4f}"
    )

    print(
        f"Learning Rate    : "
        f"{current_lr:.8f}"
    )

    print("----------------------------------------")


    # ==================================================
    # Save Checkpoint
    # ==================================================

    torch.save(

        {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "best_val_loss":
                min(
                    best_val_loss,
                    val_loss
                ),

            "train_indices":
                train_indices,

            "val_indices":
                val_indices

        },

        checkpoint_path
    )


    print(
        "Checkpoint Saved"
    )


    # ==================================================
    # Save Best Validation Model
    # ==================================================

    if val_loss < best_val_loss:

        best_val_loss = (
            val_loss
        )


        torch.save(

            model.state_dict(),

            best_model_path
        )


        print(
            "✅ Best V2 Model Updated"
        )

        print(
            f"New Best Validation Loss: "
            f"{best_val_loss:.4f}"
        )


# ======================================================
# Save Final Model
# ======================================================

torch.save(

    model.state_dict(),

    final_model_path
)


print("\n========================================")
print("V2 TRAINING COMPLETED")
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

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print("========================================")