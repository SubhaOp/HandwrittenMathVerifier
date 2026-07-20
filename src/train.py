import torch
import torch.nn as nn
from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.config import *
from src.dataset import HMEDataset
from src.model import MathRecognizer


# ======================================================
# Collate Function
# ======================================================

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


# ======================================================
# Dataset
# ======================================================

print("\nLoading Dataset...")

dataset = HMEDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn
)

print(f"Training Samples : {len(dataset)}")


# ======================================================
# Model
# ======================================================

num_classes = len(dataset.char2idx) + 1

model = MathRecognizer(num_classes).to(DEVICE)

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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ======================================================
# Checkpoint Paths
# ======================================================

checkpoint_path = MODEL_DIR / "checkpoint.pth"

best_model_path = MODEL_DIR / "best_model.pth"

final_model_path = MODEL_DIR / "math_recognizer.pth"


# ======================================================
# Resume Training
# ======================================================

start_epoch = 0

best_loss = float("inf")

if checkpoint_path.exists():

    print("\nLoading Checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_loss = checkpoint["loss"]

    print(f"Resuming from Epoch {start_epoch}")

else:

    print("\nStarting Fresh Training...")


# ======================================================
# Training
# ======================================================

model.train()

print("\nTraining Started...\n")

for epoch in range(start_epoch, EPOCHS):

    total_loss = 0.0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(images)

        outputs = outputs.log_softmax(2)

        outputs = outputs.permute(1, 0, 2)

        input_lengths = torch.full(
            (images.size(0),),
            outputs.size(0),
            dtype=torch.long,
            device=DEVICE
        )

        target_lengths = (labels != 0).sum(dim=1)

        loss = criterion(
            outputs,
            labels,
            input_lengths,
            target_lengths
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    print("\n----------------------------------------")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Average Loss : {avg_loss:.4f}")
    print("----------------------------------------")

    # ==========================================
    # Save Resume Checkpoint
    # ==========================================

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        },
        checkpoint_path
    )

    print("Checkpoint Saved")

    # ==========================================
    # Save Best Model
    # ==========================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("Best Model Updated")


# ======================================================
# Save Final Model
# ======================================================

torch.save(
    model.state_dict(),
    final_model_path
)

print("\n========================================")
print("Training Completed Successfully")
print("========================================")

print(f"Checkpoint : {checkpoint_path}")
print(f"Best Model : {best_model_path}")
print(f"Final Model: {final_model_path}")