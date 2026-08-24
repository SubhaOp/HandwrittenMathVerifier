"""
Greedy vs Beam Search Evaluation

Compares:
1. CTC Greedy decoding
2. CTC Prefix Beam Search

on the same subset of the HME100K test dataset.
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


# ======================================================
# Experiment Configuration
# ======================================================

# Start small because beam search is much slower.
# After confirming it works, increase this.
MAX_SAMPLES = 1000

BEAM_WIDTH = 10


# ======================================================
# Load Test Labels
# ======================================================

def load_test_labels():

    df = pd.read_csv(
        TEST_LABEL_FILE,
        sep="\t",
        header=None,
        names=["image", "label"]
    )

    return df


# ======================================================
# Evaluation
# ======================================================

def evaluate():

    print("\nLoading Test Dataset...")

    df = load_test_labels()

    # --------------------------------------------------
    # Use same subset for both decoders
    # --------------------------------------------------

    if MAX_SAMPLES is not None:
        df = df.iloc[:MAX_SAMPLES].copy()

    total = len(df)

    print(f"Evaluation Samples : {total}")
    print(f"Beam Width         : {BEAM_WIDTH}")

    greedy_correct = 0
    beam_correct = 0

    greedy_errors = 0
    beam_errors = 0

    greedy_total_time = 0.0
    beam_total_time = 0.0

    results = []

    # ==================================================
    # Evaluate
    # ==================================================

    for _, row in tqdm(
        df.iterrows(),
        total=total,
        desc="Comparing Decoders"
    ):

        image_path = TEST_IMAGE_DIR / row["image"]

        ground_truth = str(
            row["label"]
        ).strip()

        # ==============================================
        # Greedy Prediction
        # ==============================================

        start = time.perf_counter()

        try:

            greedy_prediction = predict(
                image_path,
                decoder="greedy"
            ).strip()

            greedy_error = ""

        except Exception as e:

            greedy_prediction = ""
            greedy_error = str(e)

            greedy_errors += 1

        greedy_time = (
            time.perf_counter() - start
        )

        greedy_total_time += greedy_time

        greedy_is_correct = (
            greedy_prediction == ground_truth
        )

        if greedy_is_correct:
            greedy_correct += 1


        # ==============================================
        # Beam Search Prediction
        # ==============================================

        start = time.perf_counter()

        try:

            beam_prediction = predict(
                image_path,
                decoder="beam",
                beam_width=BEAM_WIDTH
            ).strip()

            beam_error = ""

        except Exception as e:

            beam_prediction = ""
            beam_error = str(e)

            beam_errors += 1

        beam_time = (
            time.perf_counter() - start
        )

        beam_total_time += beam_time

        beam_is_correct = (
            beam_prediction == ground_truth
        )

        if beam_is_correct:
            beam_correct += 1


        # ==============================================
        # Save Result
        # ==============================================

        results.append({

            "image":
                row["image"],

            "ground_truth":
                ground_truth,

            "greedy_prediction":
                greedy_prediction,

            "beam_prediction":
                beam_prediction,

            "greedy_correct":
                greedy_is_correct,

            "beam_correct":
                beam_is_correct,

            "greedy_time":
                greedy_time,

            "beam_time":
                beam_time,

            "greedy_error":
                greedy_error,

            "beam_error":
                beam_error
        })


    # ==================================================
    # Metrics
    # ==================================================

    greedy_accuracy = (
        greedy_correct / total * 100
        if total else 0
    )

    beam_accuracy = (
        beam_correct / total * 100
        if total else 0
    )

    greedy_avg_time = (
        greedy_total_time / total
        if total else 0
    )

    beam_avg_time = (
        beam_total_time / total
        if total else 0
    )

    improvement = (
        beam_accuracy - greedy_accuracy
    )


    # ==================================================
    # Final Results
    # ==================================================

    print("\n")
    print("=" * 70)
    print("GREEDY VS BEAM SEARCH RESULTS")
    print("=" * 70)

    print(f"Total Images           : {total}")

    print("\n--- GREEDY CTC ---")

    print(
        f"Correct Predictions    : "
        f"{greedy_correct}"
    )

    print(
        f"Incorrect Predictions  : "
        f"{total - greedy_correct}"
    )

    print(
        f"Prediction Errors      : "
        f"{greedy_errors}"
    )

    print(
        f"Exact Match Accuracy   : "
        f"{greedy_accuracy:.2f}%"
    )

    print(
        f"Average Inference Time : "
        f"{greedy_avg_time:.4f} sec/image"
    )


    print("\n--- CTC BEAM SEARCH ---")

    print(
        f"Correct Predictions    : "
        f"{beam_correct}"
    )

    print(
        f"Incorrect Predictions  : "
        f"{total - beam_correct}"
    )

    print(
        f"Prediction Errors      : "
        f"{beam_errors}"
    )

    print(
        f"Exact Match Accuracy   : "
        f"{beam_accuracy:.2f}%"
    )

    print(
        f"Average Inference Time : "
        f"{beam_avg_time:.4f} sec/image"
    )


    print("\n--- COMPARISON ---")

    print(
        f"Greedy Accuracy        : "
        f"{greedy_accuracy:.2f}%"
    )

    print(
        f"Beam Accuracy          : "
        f"{beam_accuracy:.2f}%"
    )

    print(
        f"Difference             : "
        f"{improvement:+.2f}%"
    )

    if beam_accuracy > greedy_accuracy:

        print(
            "Result                 : "
            "✅ Beam Search Improved Accuracy"
        )

    elif beam_accuracy == greedy_accuracy:

        print(
            "Result                 : "
            "⚠️ No Accuracy Improvement"
        )

    else:

        print(
            "Result                 : "
            "❌ Beam Search Reduced Accuracy"
        )

    print("=" * 70)


    # ==================================================
    # Save CSV
    # ==================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR /
        "greedy_vs_beam_results.csv"
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nDetailed comparison saved to:"
    )

    print(output_file)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    evaluate()