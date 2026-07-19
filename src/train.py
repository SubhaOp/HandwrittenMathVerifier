import os
import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.dataset import HMEDataset
from src.model import MathRecognizer


# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
# Collate Function
# -------------------------------------------------
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


# -------------------------------------------------
# Dataset
# -------------------------------------------------
dataset = HMEDataset()

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)


# -------------------------------------------------
# Model
# -------------------------------------------------
num_classes = len(dataset.char2idx) + 1

model = MathRecognizer(num_classes).to(device)


# -------------------------------------------------
# Loss
# -------------------------------------------------
criterion = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)


# -------------------------------------------------
# Optimizer
# -------------------------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# -------------------------------------------------
# Checkpoint Folder
# -------------------------------------------------
checkpoint_dir = "saved_models"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = os.path.join(
    checkpoint_dir,
    "checkpoint.pth"
)

best_model_path = os.path.join(
    checkpoint_dir,
    "best_model.pth"
)


# -------------------------------------------------
# Resume Training
# -------------------------------------------------
start_epoch = 0
num_epochs = 20
best_loss = float("inf")

if os.path.exists(checkpoint_path):

    print("\nLoading previous checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]

    print(f"Resuming from epoch {start_epoch}")


# -------------------------------------------------
# Training
# -------------------------------------------------
model.train()

for epoch in range(start_epoch, num_epochs):

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{num_epochs}"
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        outputs = outputs.log_softmax(2)

        outputs = outputs.permute(1, 0, 2)

        input_lengths = torch.full(
            (images.size(0),),
            outputs.size(0),
            dtype=torch.long,
            device=device
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

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Average Loss : {avg_loss:.4f}")

    # ---------------------------------------------
    # Save checkpoint every epoch
    # ---------------------------------------------
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        },
        checkpoint_path
    )

    print("Checkpoint saved.")

    # ---------------------------------------------
    # Save best model
    # ---------------------------------------------
    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("Best model updated!")



# -------------------------------------------------
# Save Final Model
# -------------------------------------------------
torch.save(
    model.state_dict(),
    os.path.join(
        checkpoint_dir,
        "math_recognizer.pth"
    )
)

print("\nTraining Finished Successfully!")
print("Final model saved.")
print("Best model saved.")
print("Checkpoint saved.")