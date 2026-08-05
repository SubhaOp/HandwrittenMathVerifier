"""
Full Model Evaluation Script

Evaluates the trained handwritten mathematical expression
recognition model on the complete test dataset.
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

    correct = 0
    errors = 0
    total_inference_time = 0.0

    results = []

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
            error_message = ""

        except Exception as e:
            prediction = ""
            error_message = str(e)
            errors += 1

        elapsed = time.perf_counter() - start
        total_inference_time += elapsed

        is_correct = prediction == ground_truth

        if is_correct:
            correct += 1

        results.append({
            "image": row["image"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": is_correct,
            "inference_time": elapsed,
            "error": error_message
        })

    # -------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------

    accuracy = (correct / total) * 100 if total else 0

    avg_inference_time = (
        total_inference_time / total
        if total else 0
    )

    # -------------------------------------------------
    # Print results
    # -------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print(f"Total Test Images       : {total}")
    print(f"Correct Predictions     : {correct}")
    print(f"Incorrect Predictions   : {total - correct}")
    print(f"Prediction Errors       : {errors}")
    print(f"Exact Match Accuracy    : {accuracy:.2f}%")
    print(f"Total Inference Time    : {total_inference_time:.2f} sec")
    print(f"Average Inference Time  : {avg_inference_time:.4f} sec/image")

    print("=" * 70)

    # -------------------------------------------------
    # Save predictions
    # -------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "evaluation_results.csv"

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_file,
        index=False
    )

    print(f"\nEvaluation results saved to:")
    print(output_file)


if __name__ == "__main__":
    evaluate()