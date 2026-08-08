import json
import math
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

BLANK_ID = 0


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

#preprocess image to match training preprocessing
def preprocess(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ==================================================
    # Aspect-ratio preserving preprocessing
    # SAME preprocessing used during V2 training
    # ==================================================

    original_height, original_width = image.shape[:2]

    if original_height <= 0 or original_width <= 0:
        raise ValueError(
            f"Invalid image dimensions: "
            f"{original_width}x{original_height}"
        )

    # Calculate scale while preserving aspect ratio
    scale = min(
        IMAGE_WIDTH / original_width,
        IMAGE_HEIGHT / original_height
    )

    new_width = max(
        1,
        int(round(original_width * scale))
    )

    new_height = max(
        1,
        int(round(original_height * scale))
    )

    # Safety
    new_width = min(
        new_width,
        IMAGE_WIDTH
    )

    new_height = min(
        new_height,
        IMAGE_HEIGHT
    )

    # Same interpolation strategy as dataset.py
    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    # ==================================================
    # Create white canvas
    # ==================================================

    canvas = (
        torch.ones(
            (
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                CHANNELS
            ),
            dtype=torch.uint8
        ).numpy()
        * 255
    )

    # ==================================================
    # Vertical centering
    # Horizontal LEFT alignment
    #
    # This must match dataset.py
    # ==================================================

    y_offset = (
        IMAGE_HEIGHT - new_height
    ) // 2

    x_offset = 0

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    # ==================================================
    # Normalize
    # ==================================================

    image = (
        canvas.astype("float32")
        / 255.0
    )

    # HWC -> CHW
    image = torch.from_numpy(
        image
    ).float()

    image = image.permute(
        2,
        0,
        1
    )

    # Add batch dimension
    image = image.unsqueeze(0)

    return image.to(DEVICE)

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

def greedy_decode(indices):

    prediction = []
    previous = None

    for idx in indices:

        idx = int(idx)

        if idx == BLANK_ID:
            previous = None
            continue

        if idx == previous:
            continue

        char = idx2char.get(str(idx), "")

        prediction.append(char)

        previous = idx

    return "".join(prediction).strip()


# ======================================================
# Helper for Beam Search
# ======================================================

def log_add(*values):

    values = [v for v in values if v != -float("inf")]

    if not values:
        return -float("inf")

    maximum = max(values)

    return maximum + math.log(
        sum(math.exp(v - maximum) for v in values)
    )


# ======================================================
# CTC Prefix Beam Search
# ======================================================

def beam_search_decode(log_probs, beam_width=10, token_topk=20):

    """
    Proper CTC prefix beam search.

    log_probs shape:
        [T, C]

    T = time steps
    C = number of classes
    """

    neg_inf = -float("inf")

    # prefix -> (prob ending blank, prob ending non-blank)
    beams = {
        (): (0.0, neg_inf)
    }

    for t in range(log_probs.size(0)):

        timestep = log_probs[t]

        # Keep only the strongest classes at this timestep.
        # This makes decoding much faster.
        k = min(token_topk, timestep.size(0))

        top_values, top_indices = torch.topk(
            timestep,
            k=k
        )

        next_beams = {}

        for prefix, (p_blank, p_non_blank) in beams.items():

            for log_p, token in zip(
                top_values.tolist(),
                top_indices.tolist()
            ):

                # --------------------------------------
                # Blank
                # --------------------------------------

                if token == BLANK_ID:

                    n_blank, n_non_blank = next_beams.get(
                        prefix,
                        (neg_inf, neg_inf)
                    )

                    n_blank = log_add(
                        n_blank,
                        p_blank + log_p,
                        p_non_blank + log_p
                    )

                    next_beams[prefix] = (
                        n_blank,
                        n_non_blank
                    )

                    continue

                # --------------------------------------
                # Non-blank
                # --------------------------------------

                last_token = prefix[-1] if prefix else None

                # Same token as previous prefix token
                if token == last_token:

                    # Case 1:
                    # repeated token without blank
                    # keeps the same collapsed prefix
                    n_blank, n_non_blank = next_beams.get(
                        prefix,
                        (neg_inf, neg_inf)
                    )

                    n_non_blank = log_add(
                        n_non_blank,
                        p_non_blank + log_p
                    )

                    next_beams[prefix] = (
                        n_blank,
                        n_non_blank
                    )

                    # Case 2:
                    # repeated token after blank
                    # creates a new repeated symbol
                    new_prefix = prefix + (token,)

                    n_blank, n_non_blank = next_beams.get(
                        new_prefix,
                        (neg_inf, neg_inf)
                    )

                    n_non_blank = log_add(
                        n_non_blank,
                        p_blank + log_p
                    )

                    next_beams[new_prefix] = (
                        n_blank,
                        n_non_blank
                    )

                else:

                    new_prefix = prefix + (token,)

                    n_blank, n_non_blank = next_beams.get(
                        new_prefix,
                        (neg_inf, neg_inf)
                    )

                    n_non_blank = log_add(
                        n_non_blank,
                        p_blank + log_p,
                        p_non_blank + log_p
                    )

                    next_beams[new_prefix] = (
                        n_blank,
                        n_non_blank
                    )

        # ----------------------------------------------
        # Keep strongest prefixes
        # ----------------------------------------------

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: log_add(
                    item[1][0],
                    item[1][1]
                ),
                reverse=True
            )[:beam_width]
        )

    # ==================================================
    # Select best final prefix
    # ==================================================

    best_prefix = max(
        beams.items(),
        key=lambda item: log_add(
            item[1][0],
            item[1][1]
        )
    )[0]

    characters = [
        idx2char.get(str(idx), "")
        for idx in best_prefix
    ]

    return "".join(characters).strip()


# ======================================================
# Prediction
# ======================================================

def predict(
    image_path,
    decoder="greedy",
    beam_width=10,
    token_topk=20
):

    image = preprocess(image_path)

    with torch.inference_mode():

        output = model(image)

        # ----------------------------------------------
        # Greedy CTC
        # ----------------------------------------------

        if decoder == "greedy":

            prediction = output.argmax(dim=2)
            prediction = prediction.squeeze(0)

            return greedy_decode(prediction)

        # ----------------------------------------------
        # Beam Search CTC
        # ----------------------------------------------

        elif decoder == "beam":

            log_probs = torch.log_softmax(
                output.squeeze(0),
                dim=-1
            )

            return beam_search_decode(
                log_probs.cpu(),
                beam_width=beam_width,
                token_topk=token_topk
            )

        else:

            raise ValueError(
                f"Unknown decoder: {decoder}"
            )


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    image_path = input("Image Path: ").strip()

    print("\nRunning Greedy Decoder...")

    greedy_prediction = predict(
        image_path,
        decoder="greedy"
    )

    print("\nGreedy Prediction:")
    print(greedy_prediction)

    print("\nRunning Beam Search Decoder...")

    beam_prediction = predict(
        image_path,
        decoder="beam",
        beam_width=10
    )

    print("\nBeam Search Prediction:")
    print(beam_prediction)