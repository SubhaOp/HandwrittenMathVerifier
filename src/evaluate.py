"""
Model Evaluation Script (Debug Version)

This version prints the first prediction and exits.
"""

import time
import pandas as pd
from tqdm import tqdm

from src.predict import predict
from src.config import (
    TEST_IMAGE_DIR,
    TEST_LABEL_FILE,
    OUTPUT_DIR
)


def load_test_labels():

    df = pd.read_csv(
        TEST_LABEL_FILE,
        sep="\t",
        header=None,
        names=["image", "label"]
    )

    return df


def evaluate():

    print("\nLoading Test Dataset...")

    df = load_test_labels()

    total = len(df)

    print(f"Total Test Images : {total}\n")

    for _, row in tqdm(
        df.iterrows(),
        total=total,
        desc="Evaluating"
    ):

        image_path = TEST_IMAGE_DIR / row["image"]

        ground_truth = str(row["label"]).strip()

        start = time.perf_counter()

        try:
            prediction = predict(image_path).strip()
        except Exception as e:
            prediction = f"ERROR: {e}"

        elapsed = time.perf_counter() - start

        print("\n")
        print("=" * 70)
        print("Image File    :", row["image"])
        print("Ground Truth  :", repr(ground_truth))
        print("Prediction    :", repr(prediction))
        print(f"Inference Time: {elapsed:.4f} sec")
        print("=" * 70)

        if prediction == ground_truth:
            print("\n✅ FIRST SAMPLE MATCHES")
        else:
            print("\n❌ FIRST SAMPLE DOES NOT MATCH")

        # Stop after first image
        return


if __name__ == "__main__":
    evaluate()