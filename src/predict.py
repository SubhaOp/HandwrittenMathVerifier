import json
import cv2
import torch

from src.config import *
from src.model import MathRecognizer

# ======================================================
# Load Vocabulary
# ======================================================

with open(CHAR2IDX_FILE, "r", encoding="utf-8") as f:
    char2idx = json.load(f)

with open(IDX2CHAR_FILE, "r", encoding="utf-8") as f:
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

    image = cv2.imread(str(image_path))

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
# CTC Greedy Decoder
# ======================================================

def decode(indices):

    prediction = []

    previous = None

    for idx in indices:

        idx = int(idx)

        # Ignore blank token
        if idx == 0:
            previous = None
            continue

        # Collapse repeated characters
        if idx == previous:
            continue

        char = idx2char.get(str(idx), "")

        prediction.append(char)

        previous = idx

    text = "".join(prediction)

    # remove leading/trailing whitespace only
    text = text.strip()

    return text


# ======================================================
# Prediction
# ======================================================

def predict(image_path):

    image = preprocess(image_path)

    with torch.inference_mode():

        output = model(image)

        prediction = output.argmax(dim=2)

        prediction = prediction.squeeze(0)

    return decode(prediction)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    image_path = input("Image Path: ").strip()

    prediction = predict(image_path)

    print("\nPredicted Expression:")
    print(prediction)