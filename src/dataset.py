import json
from pathlib import Path

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

        with open(CHAR2IDX_FILE, "r") as f:
            self.char2idx = json.load(f)

    def __len__(self):
        return len(self.df)

    def encode_label(self, text):

        return torch.tensor(
            [self.char2idx[c] for c in text],
            dtype=torch.long
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.image_dir / row["image"]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        image = image.astype("float32") / 255.0

        image = torch.tensor(image, dtype=torch.float32)

        # Convert HWC -> CHW
        image = image.permute(2, 0, 1)

        label = self.encode_label(row["label"])

        
        return image, label
        