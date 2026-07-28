# Prediction entrypoint placeholder
import json
import cv2
import torch

from pathlib import Path

from src.config import *
from src.model import MathRecognizer


# ======================================================
# Load Vocabulary
# ======================================================

with open(CHAR2IDX_FILE, "r") as f:
    char2idx = json.load(f)

with open(IDX2CHAR_FILE, "r") as f:
    idx2char = json.load(f)

num_classes = len(char2idx) + 1


# ======================================================
# Load Model
# ======================================================

model = MathRecognizer(num_classes)

checkpoint = torch.load(
    MODEL_DIR / "best_model.pth",
    map_location=DEVICE
)

# Supports both state_dict-only and checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("✅ Model Loaded Successfully")


# ======================================================
# Image Preprocessing
# ======================================================

def preprocess(image_path):

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    image = image.astype("float32") / 255.0

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    image = image.permute(2, 0, 1)

    image = image.unsqueeze(0)

    return image.to(DEVICE)


# ======================================================
# Greedy CTC Decoder
# ======================================================

def decode(indices):

    prediction = []

    previous = -1

    for idx in indices:

        idx = int(idx)

        if idx == 0:
            previous = idx
            continue

        if idx == previous:
            continue

        prediction.append(
            idx2char[str(idx)]
        )

        previous = idx

    return "".join(prediction)


# ======================================================
# Prediction
# ======================================================

def predict(image_path):

    image = preprocess(image_path)

    with torch.no_grad():

        output = model(image)

        output = torch.softmax(
            output,
            dim=2
        )

        prediction = output.argmax(2)

        prediction = prediction.squeeze(0)

    return decode(prediction)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    image_path = input("Image Path : ").strip()

    prediction = predict(image_path)

    print("\nPrediction:")
    print(prediction)