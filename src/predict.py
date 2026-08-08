import json
import math

import cv2
import torch

from src.config import *
from src.model import MathRecognizer


# ======================================================
# V3 Configuration
# ======================================================

MODEL_FILENAME = "best_model_v3.pth"

DEFAULT_BEAM_WIDTH = 10

BLANK_ID = 0


# ======================================================
# Load Vocabulary
# ======================================================

with open(
    CHAR2IDX_FILE,
    "r",
    encoding="utf-8"
) as f:

    char2idx = json.load(f)


with open(
    IDX2CHAR_FILE,
    "r",
    encoding="utf-8"
) as f:

    idx2char = json.load(f)


num_classes = (
    len(char2idx) + 1
)


# ======================================================
# Load V3 Model
# ======================================================

model = MathRecognizer(
    num_classes
)


model_path = (
    MODEL_DIR /
    MODEL_FILENAME
)


if not model_path.exists():

    raise FileNotFoundError(

        f"\nV3 model not found:\n"
        f"{model_path}\n\n"

        f"Train V3 first and make sure "
        f"best_model_v3.pth exists."
    )


checkpoint = torch.load(

    model_path,

    map_location=DEVICE
)


# Supports:
#
# 1. state_dict-only model
# 2. checkpoint containing model_state_dict

if (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
):

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(
    DEVICE
)

model.eval()


print(
    "========================================"
)

print(
    "V3 Model Loaded Successfully"
)

print(
    f"Model : {model_path}"
)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU   : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    "========================================"
)


# ======================================================
# Image Preprocessing
# ======================================================

def preprocess(image_path):

    """
    IMPORTANT:

    This preprocessing MUST match dataset.py.

    V3 uses:
        - RGB image
        - aspect-ratio preserving resize
        - white padding
        - vertical centering
        - horizontal left alignment
        - normalization /255
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )


    if image is None:

        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )


    # --------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------
    # Original dimensions
    # --------------------------------------------------

    original_height, original_width = (
        image.shape[:2]
    )


    if (
        original_height <= 0
        or
        original_width <= 0
    ):

        raise ValueError(

            "Invalid image dimensions: "
            f"{original_width}x"
            f"{original_height}"
        )


    # ==================================================
    # Aspect-ratio preserving resize
    # ==================================================

    scale = min(

        IMAGE_WIDTH / original_width,

        IMAGE_HEIGHT / original_height
    )


    new_width = max(

        1,

        int(
            round(
                original_width * scale
            )
        )
    )


    new_height = max(

        1,

        int(
            round(
                original_height * scale
            )
        )
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


    # --------------------------------------------------
    # Interpolation
    # --------------------------------------------------

    interpolation = (

        cv2.INTER_AREA

        if scale < 1.0

        else cv2.INTER_CUBIC
    )


    resized = cv2.resize(

        image,

        (
            new_width,
            new_height
        ),

        interpolation=interpolation
    )


    # ==================================================
    # White canvas
    # ==================================================

    canvas = (

        torch.ones(

            (
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                CHANNELS
            ),

            dtype=torch.uint8
        )

        * 255

    ).numpy()


    # ==================================================
    # Center vertically
    # Left-align horizontally
    # ==================================================

    y_offset = (

        IMAGE_HEIGHT
        -
        new_height
    ) // 2


    x_offset = 0


    canvas[

        y_offset:
        y_offset + new_height,

        x_offset:
        x_offset + new_width

    ] = resized


    # ==================================================
    # Normalize
    # ==================================================

    image = (

        canvas.astype(
            "float32"
        )
        /
        255.0
    )


    # NumPy -> Tensor

    image = torch.from_numpy(
        image
    ).float()


    # HWC -> CHW

    image = image.permute(
        2,
        0,
        1
    )


    # Add batch dimension

    image = image.unsqueeze(
        0
    )


    return image.to(
        DEVICE
    )


# ======================================================
# CTC Greedy Decoder
# ======================================================

def decode(indices):

    """
    Standard CTC greedy decoding.

    Steps:
        1. Remove blank tokens.
        2. Collapse consecutive duplicate tokens.
        3. Convert token IDs to characters.
    """

    prediction = []

    previous = None


    for idx in indices:

        idx = int(idx)


        # --------------------------------------------------
        # Blank
        # --------------------------------------------------

        if idx == BLANK_ID:

            previous = None

            continue


        # --------------------------------------------------
        # Repeated token
        # --------------------------------------------------

        if idx == previous:

            continue


        # --------------------------------------------------
        # Convert ID -> character
        # --------------------------------------------------

        char = idx2char.get(

            str(idx),

            ""
        )


        if char != "":

            prediction.append(
                char
            )


        previous = idx


    text = "".join(
        prediction
    )


    return text.strip()


# ======================================================
# Log Add
# ======================================================

def log_add(a, b):

    """
    Stable log(a + b).
    """

    if a == -math.inf:

        return b


    if b == -math.inf:

        return a


    if a < b:

        a, b = b, a


    return (

        a
        +
        math.log1p(
            math.exp(
                b - a
            )
        )
    )


# ======================================================
# CTC Prefix Beam Search
# ======================================================

def ctc_prefix_beam_search(
    log_probs,
    beam_width=10
):

    """
    Prefix Beam Search for CTC.

    log_probs:
        Tensor of shape (T, C)

    T:
        Number of CTC time steps.

    C:
        Number of output classes.

    Returns:
        decoded character string
    """

    # --------------------------------------------------
    # Move to CPU
    # --------------------------------------------------

    log_probs = (
        log_probs
        .detach()
        .cpu()
    )


    time_steps = (
        log_probs.shape[0]
    )


    # --------------------------------------------------
    # Each prefix has:
    #
    # (probability ending in blank,
    #  probability ending in non-blank)
    # --------------------------------------------------

    beams = {

        (): (
            0.0,
            -math.inf
        )

    }


    # ==================================================
    # Process each time step
    # ==================================================

    for t in range(
        time_steps
    ):

        next_beams = {}


        # --------------------------------------------------
        # Limit classes considered at each timestep.
        #
        # This speeds up beam search substantially.
        # --------------------------------------------------

        class_count = log_probs.shape[1]

        top_k = min(
            class_count,
            max(
                beam_width * 2,
                20
            )
        )


        top_values, top_indices = torch.topk(

            log_probs[t],

            k=top_k
        )


        # --------------------------------------------------
        # Current beams
        # --------------------------------------------------

        for prefix, (
            prob_blank,
            prob_nonblank
        ) in beams.items():


            # ==============================================
            # Probability of prefix before current timestep
            # ==============================================

            total_prefix_prob = log_add(

                prob_blank,

                prob_nonblank
            )


            # ==============================================
            # Process likely classes
            # ==============================================

            for log_p, class_idx in zip(

                top_values.tolist(),

                top_indices.tolist()
            ):

                class_idx = int(
                    class_idx
                )


                # ------------------------------------------
                # Blank
                # ------------------------------------------

                if class_idx == BLANK_ID:

                    old_blank, old_nonblank = (

                        next_beams.get(

                            prefix,

                            (
                                -math.inf,
                                -math.inf
                            )
                        )
                    )


                    new_blank = log_add(

                        old_blank,

                        total_prefix_prob
                        +
                        log_p
                    )


                    next_beams[prefix] = (

                        new_blank,

                        old_nonblank
                    )


                    continue


                # ------------------------------------------
                # Character
                # ------------------------------------------

                char = idx2char.get(

                    str(class_idx),

                    ""
                )


                if char == "":

                    continue


                # ------------------------------------------
                # Case 1:
                #
                # Same character as last character
                # ------------------------------------------

                if (

                    len(prefix) > 0

                    and

                    prefix[-1] == class_idx

                ):

                    # Repeating the same character
                    # without a blank does NOT extend
                    # the CTC prefix.

                    old_blank, old_nonblank = (

                        next_beams.get(

                            prefix,

                            (
                                -math.inf,
                                -math.inf
                            )
                        )
                    )


                    new_nonblank = log_add(

                        old_nonblank,

                        prob_nonblank
                        +
                        log_p
                    )


                    # A repeated character can also
                    # come from the blank state and
                    # create a new character.

                    new_prefix = (

                        prefix
                        +
                        (
                            class_idx,
                        )
                    )


                    np_blank, np_nonblank = (

                        next_beams.get(

                            new_prefix,

                            (
                                -math.inf,
                                -math.inf
                            )
                        )
                    )


                    np_nonblank = log_add(

                        np_nonblank,

                        prob_blank
                        +
                        log_p
                    )


                    next_beams[prefix] = (

                        old_blank,

                        new_nonblank
                    )


                    next_beams[new_prefix] = (

                        np_blank,

                        np_nonblank
                    )


                else:

                    # --------------------------------------
                    # Normal character extension
                    # --------------------------------------

                    new_prefix = (

                        prefix
                        +
                        (
                            class_idx,
                        )
                    )


                    old_blank, old_nonblank = (

                        next_beams.get(

                            new_prefix,

                            (
                                -math.inf,
                                -math.inf
                            )
                        )
                    )


                    new_nonblank = log_add(

                        old_nonblank,

                        total_prefix_prob
                        +
                        log_p
                    )


                    next_beams[new_prefix] = (

                        old_blank,

                        new_nonblank
                    )


        # --------------------------------------------------
        # Keep only strongest beams
        # --------------------------------------------------

        beams = dict(

            sorted(

                next_beams.items(),

                key=lambda item:
                    log_add(
                        item[1][0],
                        item[1][1]
                    ),

                reverse=True

            )[
                :beam_width
            ]
        )


    # ==================================================
    # Best Prefix
    # ==================================================

    if not beams:

        return ""


    best_prefix = max(

        beams.keys(),

        key=lambda prefix:

            log_add(

                beams[prefix][0],

                beams[prefix][1]
            )
    )


    # ==================================================
    # Convert IDs -> characters
    # ==================================================

    result = []


    for idx in best_prefix:

        char = idx2char.get(

            str(idx),

            ""
        )


        if char != "":

            result.append(
                char
            )


    return "".join(
        result
    ).strip()


# ======================================================
# Greedy Prediction
# ======================================================

def predict_greedy(
    image
):

    with torch.inference_mode():

        output = model(
            image
        )


        # B,T,C -> T,C

        log_probs = torch.log_softmax(

            output,

            dim=2
        ).squeeze(0)


        indices = torch.argmax(

            log_probs,

            dim=1
        )


    return decode(
        indices
    )


# ======================================================
# Beam Prediction
# ======================================================

def predict_beam(
    image,
    beam_width=DEFAULT_BEAM_WIDTH
):

    with torch.inference_mode():

        output = model(
            image
        )


        # B,T,C -> T,C

        log_probs = torch.log_softmax(

            output,

            dim=2
        ).squeeze(0)


    return ctc_prefix_beam_search(

        log_probs,

        beam_width=beam_width
    )


# ======================================================
# Main Prediction Function
# ======================================================

def predict(

    image_path,

    decoder="greedy",

    beam_width=DEFAULT_BEAM_WIDTH

):

    """
    Main prediction function.

    decoder:
        "greedy"
        "beam"

    Example:

        predict(
            image_path,
            decoder="greedy"
        )

        predict(
            image_path,
            decoder="beam",
            beam_width=10
        )
    """

    image = preprocess(
        image_path
    )


    if decoder == "greedy":

        return predict_greedy(
            image
        )


    elif decoder == "beam":

        return predict_beam(

            image,

            beam_width=beam_width
        )


    else:

        raise ValueError(

            "Unknown decoder: "
            f"{decoder}. "

            "Use 'greedy' or 'beam'."
        )


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "Handwritten Mathematical Expression"
    )

    print(
        "V3 Predictor"
    )

    print(
        "========================================"
    )


    image_path = input(
        "\nImage Path: "
    ).strip()


    print(
        "\nChoose decoder:"
    )

    print(
        "1. Greedy"
    )

    print(
        "2. Beam Search"
    )


    choice = input(
        "Choice [1/2]: "
    ).strip()


    if choice == "2":

        beam_width_input = input(

            "Beam width [10]: "
        ).strip()


        if beam_width_input == "":

            beam_width = 10

        else:

            beam_width = int(
                beam_width_input
            )


        prediction = predict(

            image_path,

            decoder="beam",

            beam_width=beam_width
        )


        decoder_name = (
            f"Beam Search "
            f"(width={beam_width})"
        )


    else:

        prediction = predict(

            image_path,

            decoder="greedy"
        )


        decoder_name = "Greedy CTC"


    print(
        "\n========================================"
    )

    print(
        f"Decoder: {decoder_name}"
    )

    print(
        "Predicted Expression:"
    )

    print(
        prediction
    )

    print(
        "========================================"
    )