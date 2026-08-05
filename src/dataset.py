import json

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import *


class HMEDataset(Dataset):

    def __init__(self):

        self.image_dir = TRAIN_IMAGE_DIR
        self.label_file = TRAIN_LABEL_FILE

        self.df = pd.read_csv(
            self.label_file,
            sep="\t",
            header=None,
            names=["image", "label"]
        )

        with open(
            CHAR2IDX_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            self.char2idx = json.load(f)

    # ==================================================
    # Dataset Length
    # ==================================================

    def __len__(self):

        return len(self.df)

    # ==================================================
    # Encode Label
    # ==================================================

    def encode_label(self, text):

        encoded = []

        for ch in str(text):

            if ch not in self.char2idx:

                raise ValueError(
                    f"Character {repr(ch)} "
                    f"not found in vocabulary."
                )

            encoded.append(
                self.char2idx[ch]
            )

        return torch.tensor(
            encoded,
            dtype=torch.long
        )

    # ==================================================
    # Resize while preserving aspect ratio
    # ==================================================

    def resize_with_padding(self, image):

        """
        Resize image without stretching the handwritten
        mathematical expression.

        Final image size:
            IMAGE_HEIGHT x IMAGE_WIDTH

        The aspect ratio is preserved and unused space
        is filled with white pixels.
        """

        original_height, original_width = image.shape[:2]

        if original_height <= 0 or original_width <= 0:

            raise ValueError(
                "Invalid image dimensions: "
                f"{original_width}x{original_height}"
            )

        # ----------------------------------------------
        # Calculate scale
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Resize
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Create white canvas
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Center vertically
        #
        # Keep image LEFT aligned horizontally.
        # This is useful because CTC reads features
        # from left to right.
        # ----------------------------------------------

        y_offset = (
            IMAGE_HEIGHT - new_height
        ) // 2

        x_offset = 0

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized

        return canvas

    # ==================================================
    # Get Item
    # ==================================================

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = (
            self.image_dir /
            row["image"]
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise FileNotFoundError(
                f"Cannot read image: {image_path}"
            )

        # BGR -> RGB

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------
        # Aspect-ratio preserving preprocessing
        # ----------------------------------------------

        image = self.resize_with_padding(
            image
        )

        # ----------------------------------------------
        # Normalize
        # ----------------------------------------------

        image = (
            image.astype("float32")
            / 255.0
        )

        image = torch.from_numpy(
            image
        ).float()

        # HWC -> CHW

        image = image.permute(
            2,
            0,
            1
        )

        # ----------------------------------------------
        # Label
        # ----------------------------------------------

        label = self.encode_label(
            row["label"]
        )

        return image, label