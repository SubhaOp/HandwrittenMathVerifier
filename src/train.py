import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
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

# Number of time steps produced by the current model.
# Your model test confirmed this is 192.
CTC_TIME_STEPS = 192


# ======================================================
# Find CTC-valid samples
# ======================================================

print("\nChecking CTC validity...")

valid_indices = []
invalid_indices = []

for idx in tqdm(range(len(dataset)), desc="Checking labels"):

    # We only need the label here.
    # Avoid loading/processing the image during this scan.
    row = dataset.df.iloc[idx]

    label = dataset.encode_label(row["label"])

    target_length = len(label)

    # CTC needs an extra time step whenever two
    # consecutive target tokens are identical.
    if target_length > 1:
        repeats = int((label[1:] == label[:-1]).sum().item())
    else:
        repeats = 0

    required_steps = target_length + repeats

    if required_steps <= CTC_TIME_STEPS:
        valid_indices.append(idx)
    else:
        invalid_indices.append(idx)


print("\n========================================")
print("CTC Dataset Filtering")
print("========================================")
print(f"Original Samples : {len(dataset)}")
print(f"Valid Samples    : {len(valid_indices)}")
print(f"Filtered Samples : {len(invalid_indices)}")
print("========================================")


# ======================================================
# Create filtered dataset
# ======================================================

train_dataset = Subset(
    dataset,
    valid_indices
)


# ======================================================
# DataLoader
# ======================================================

loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn
)

print(f"Training Samples : {len(train_dataset)}")


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

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
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

print("\nTraining Started...\n")

for epoch in range(start_epoch, EPOCHS):

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        outputs = outputs.log_softmax(dim=2)

        outputs = outputs.permute(1, 0, 2)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=outputs.size(0),
            dtype=torch.long,
            device=DEVICE
        )

        target_lengths = torch.tensor(
            [torch.count_nonzero(label).item() for label in labels],
            dtype=torch.long,
            device=DEVICE
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
            max_norm=5.0
        )

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    print("\n----------------------------------------")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Average Loss : {avg_loss:.4f}")
    print("----------------------------------------")

    # ==========================================
    # Save Checkpoint
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