"""
V3 FINAL TRAINING -- CLEANED HME100K
====================================

Final training after the cleaned-dataset epoch study.

Epoch study result:
    Best validation epoch = 12
    Best validation CTC loss = 0.2961

This script:
    - uses the cleaned HME100K training set
    - expects exactly 74,226 training samples
    - uses all CTC-valid samples for final training (no 90/10 validation split)
    - trains exactly 12 epochs
    - keeps the V3 ResNet18 + BiLSTM + Linear + CTC architecture
    - keeps batch size, learning rate, CTC time steps and AdamW settings
    - saves the final model to Google Drive
    - saves a checkpoint after every epoch so training can be resumed

IMPORTANT:
    74,226 is the CLEANED TRAINING SET, not the independent test set.
    The independent test set is evaluated separately after this script.
"""

import json
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

FINAL_EPOCHS = 12
EXPECTED_CLEAN_TRAIN_COUNT = 74226

CTC_TIME_STEPS = 384

# Keep the same training settings used by the V3 epoch study.
BATCH_SIZE_FINAL = BATCH_SIZE
FINAL_LEARNING_RATE = LEARNING_RATE
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0

# Google Drive output directory
FINAL_DIR = Path(
    "/content/drive/MyDrive/HandwrittenMathVerifier/"
    "final_training_12_epoch"
)

FINAL_MODEL_PATH = FINAL_DIR / "math_recognizer_v3_final_epoch12.pth"
LATEST_CHECKPOINT_PATH = FINAL_DIR / "latest_checkpoint_epoch12.pth"

FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# DEVICE / SEED
# ======================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


print("\n" + "=" * 70)
print("V3 FINAL TRAINING -- CLEANED HME100K")
print("=" * 70)
print("Device       :", DEVICE)

if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))

print("Dataset      :", DATASET_DIR)
print("Final epochs :", FINAL_EPOCHS)
print("CTC steps    :", CTC_TIME_STEPS)
print("Output dir   :", FINAL_DIR)
print("=" * 70)


# ======================================================
# DATASET
# ======================================================

print("\nLoading cleaned training dataset...")

dataset = HMEDataset()

print("Dataset samples loaded :", len(dataset))

if len(dataset) != EXPECTED_CLEAN_TRAIN_COUNT:
    raise RuntimeError(
        f"\nExpected exactly {EXPECTED_CLEAN_TRAIN_COUNT} cleaned training "
        f"samples, but HMEDataset() loaded {len(dataset)}.\n\n"
        f"Dataset path currently used:\n{DATASET_DIR}\n\n"
        "STOPPING before training. Fix the dataset path/count first."
    )

print(
    f"Confirmed: cleaned training dataset contains "
    f"{EXPECTED_CLEAN_TRAIN_COUNT} samples."
)


# ======================================================
# CTC VALIDITY CHECK
# ======================================================

print("\nChecking CTC validity...")

valid_indices = []
invalid_indices = []

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

print("\n" + "=" * 70)
print("FINAL CTC VALIDITY CHECK")
print("=" * 70)
print("Original cleaned samples :", len(dataset))
print("CTC-valid samples        :", len(valid_indices))
print("CTC-filtered samples     :", len(invalid_indices))
print("=" * 70)

if len(invalid_indices) > 0:
    print(
        "\nWARNING: Some cleaned samples exceed the 384-step CTC limit."
    )
    print(
        "They will NOT be used because the V3 architecture cannot "
        "represent their target sequence within 384 CTC steps."
    )

if len(valid_indices) == 0:
    raise RuntimeError("No CTC-valid training samples were found.")


# ======================================================
# FINAL TRAINING DATA
# ======================================================
#
# IMPORTANT:
# No validation split here.
# Every CTC-valid cleaned training sample is used for training.
# ======================================================

train_dataset = torch.utils.data.Subset(
    dataset,
    valid_indices
)


def collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    images = torch.stack(images)

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=0
    )

    return images, labels


def get_target_lengths(labels):
    return torch.tensor(
        [
            torch.count_nonzero(label).item()
            for label in labels
        ],
        dtype=torch.long,
        device=labels.device
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_FINAL,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

print("\n" + "=" * 70)
print("FINAL TRAINING DATA")
print("=" * 70)
print("Training samples used :", len(train_dataset))
print("Batch size            :", BATCH_SIZE_FINAL)
print("Batches per epoch     :", len(train_loader))
print("Validation split      : NONE")
print("=" * 70)


# ======================================================
# MODEL
# ======================================================

num_classes = len(dataset.char2idx) + 1

print("\nVocabulary classes :", num_classes)

if len(dataset.char2idx) != 244:
    print(
        "\nWARNING: Expected 244 non-blank vocabulary symbols "
        f"for a 245-class CTC output, but found {len(dataset.char2idx)}."
    )

model = MathRecognizer(num_classes).to(DEVICE)


# ======================================================
# VERIFY MODEL OUTPUT
# ======================================================

with torch.no_grad():

    test_input = torch.zeros(
        1,
        CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        device=DEVICE
    )

    test_output = model(test_input)

if test_output.shape[1] != CTC_TIME_STEPS:

    raise RuntimeError(
        f"CTC sequence mismatch: expected {CTC_TIME_STEPS}, "
        f"got {test_output.shape[1]}"
    )

if test_output.shape[2] != num_classes:

    raise RuntimeError(
        f"Class mismatch: expected {num_classes}, "
        f"got {test_output.shape[2]}"
    )

print(
    "\nModel output verified:",
    tuple(test_output.shape)
)

del test_input
del test_output

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================
# LOSS / OPTIMIZER
# ======================================================

criterion = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=FINAL_LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ======================================================
# TRAIN ONE EPOCH
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

        outputs = model(images)

        outputs = outputs.log_softmax(
            dim=2
        )

        # B,T,C -> T,B,C
        outputs = outputs.permute(
            1,
            0,
            2
        )

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=outputs.size(0),
            dtype=torch.long,
            device=DEVICE
        )

        target_lengths = get_target_lengths(
            labels
        )

        loss = criterion(
            outputs,
            labels,
            input_lengths,
            target_lengths
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM
        )

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(train_loader)


# ======================================================
# RESUME SUPPORT
# ======================================================

start_epoch = 0

if LATEST_CHECKPOINT_PATH.exists():

    print(
        "\nCheckpoint found:"
    )

    print(
        LATEST_CHECKPOINT_PATH
    )

    response = input(
        "\nResume this 12-epoch final training? "
        "[y/N]: "
    ).strip().lower()

    if response == "y":

        checkpoint = torch.load(
            LATEST_CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        print(
            f"\nResuming from epoch {start_epoch + 1}."
        )

    else:

        print(
            "\nStarting a fresh 12-epoch final training."
        )

else:

    print(
        "\nNo final-training checkpoint found."
    )

    print(
        "Starting fresh."
    )


# ======================================================
# MAIN TRAINING LOOP
# ======================================================

if start_epoch >= FINAL_EPOCHS:

    print(
        f"\nFinal training is already complete: "
        f"{start_epoch}/{FINAL_EPOCHS} epochs."
    )

else:

    for epoch in range(
        start_epoch,
        FINAL_EPOCHS
    ):

        print("\n" + "=" * 70)
        print(
            f"FINAL TRAINING -- EPOCH "
            f"{epoch + 1}/{FINAL_EPOCHS}"
        )
        print("=" * 70)

        train_loss = train_one_epoch()

        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "-" * 70)
        print(
            f"Epoch       : {epoch + 1}/{FINAL_EPOCHS}"
        )
        print(
            f"Training CTC loss : {train_loss:.6f}"
        )
        print(
            f"Learning rate     : {current_lr:.8f}"
        )
        print("-" * 70)

        # Save checkpoint after EVERY epoch.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "final_epochs": FINAL_EPOCHS,
                "ctc_time_steps": CTC_TIME_STEPS,
                "num_classes": num_classes,
                "dataset_samples": len(dataset),
                "ctc_valid_samples": len(valid_indices),
            },
            LATEST_CHECKPOINT_PATH
        )

        print(
            "\nCheckpoint saved:"
        )

        print(
            LATEST_CHECKPOINT_PATH
        )


# ======================================================
# SAVE FINAL MODEL
# ======================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)

print("\n" + "=" * 70)
print("FINAL TRAINING COMPLETE")
print("=" * 70)
print(
    f"Epochs completed       : {FINAL_EPOCHS}"
)
print(
    f"Cleaned dataset       : {len(dataset)}"
)
print(
    f"CTC-valid samples     : {len(valid_indices)}"
)
print(
    f"Vocabulary symbols    : {len(dataset.char2idx)}"
)
print(
    f"CTC output classes    : {num_classes}"
)
print(
    f"CTC time steps        : {CTC_TIME_STEPS}"
)
print(
    f"Final model saved to  : {FINAL_MODEL_PATH}"
)
print("=" * 70)

print(
    "\nNEXT STEP:"
)
print(
    "Evaluate this final model on the independent CLEANED HME100K test set."
)
