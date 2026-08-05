#Detailed baseline performance analysis
#predict() with the greedy decoder and calculate Exact Match Accuracy + Token Error Rate + Token Accuracy + Edit Distance:
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
# Configuration
# ======================================================

# Start with 1000 to verify everything.
# Set to None later for all 24,607 test images.
MAX_SAMPLES = 1000


# ======================================================
# Tokenization
# ======================================================

def tokenize(expression):
    """
    Tokenize HME LaTeX expression.

    HME100K labels are space-separated tokens,
    so splitting on whitespace is appropriate here.
    """
    return str(expression).strip().split()


# ======================================================
# Levenshtein Edit Distance
# ======================================================

def edit_distance(reference, hypothesis):

    n = len(reference)
    m = len(hypothesis)

    previous = list(range(m + 1))

    for i in range(1, n + 1):

        current = [i] + [0] * m

        for j in range(1, m + 1):

            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1

            current[j] = min(
                previous[j] + 1,       # deletion
                current[j - 1] + 1,    # insertion
                previous[j - 1] + cost # substitution
            )

        previous = current

    return previous[m]


# ======================================================
# Load Test Dataset
# ======================================================

def load_test_labels():

    return pd.read_csv(
        TEST_LABEL_FILE,
        sep="\t",
        header=None,
        names=["image", "label"]
    )


# ======================================================
# Evaluation
# ======================================================

def evaluate():

    print("\nLoading Test Dataset...")

    df = load_test_labels()

    if MAX_SAMPLES is not None:
        df = df.iloc[:MAX_SAMPLES].copy()

    total = len(df)

    print(f"Evaluation Samples : {total}")
    print("Decoder            : Greedy CTC")

    exact_correct = 0
    prediction_errors = 0

    total_edit_distance = 0
    total_reference_tokens = 0

    total_time = 0.0

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

            prediction = predict(
                image_path,
                decoder="greedy"
            ).strip()

            error = ""

        except Exception as e:

            prediction = ""
            error = str(e)
            prediction_errors += 1

        elapsed = time.perf_counter() - start
        total_time += elapsed

        # ----------------------------------------------
        # Exact Match
        # ----------------------------------------------

        exact_match = prediction == ground_truth

        if exact_match:
            exact_correct += 1

        # ----------------------------------------------
        # Token Metrics
        # ----------------------------------------------

        reference_tokens = tokenize(ground_truth)
        predicted_tokens = tokenize(prediction)

        distance = edit_distance(
            reference_tokens,
            predicted_tokens
        )

        total_edit_distance += distance
        total_reference_tokens += len(reference_tokens)

        if len(reference_tokens) > 0:

            sample_ter = (
                distance / len(reference_tokens)
            )

        else:

            sample_ter = 0.0 if len(predicted_tokens) == 0 else 1.0

        results.append({

            "image":
                row["image"],

            "ground_truth":
                ground_truth,

            "prediction":
                prediction,

            "exact_match":
                exact_match,

            "reference_tokens":
                len(reference_tokens),

            "predicted_tokens":
                len(predicted_tokens),

            "edit_distance":
                distance,

            "token_error_rate":
                sample_ter,

            "inference_time":
                elapsed,

            "error":
                error
        })


    # ==================================================
    # Final Metrics
    # ==================================================

    exact_accuracy = (
        exact_correct / total * 100
        if total else 0
    )

    token_error_rate = (
        total_edit_distance /
        total_reference_tokens
        if total_reference_tokens else 0
    )

    token_accuracy = max(
        0.0,
        1.0 - token_error_rate
    )

    avg_edit_distance = (
        total_edit_distance / total
        if total else 0
    )

    avg_time = (
        total_time / total
        if total else 0
    )


    # ==================================================
    # Print Results
    # ==================================================

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    print(f"Total Images              : {total}")
    print(f"Correct Exact Matches     : {exact_correct}")
    print(f"Incorrect Exact Matches   : {total - exact_correct}")
    print(f"Prediction Errors         : {prediction_errors}")

    print("-" * 70)

    print(
        f"Exact Match Accuracy      : "
        f"{exact_accuracy:.2f}%"
    )

    print(
        f"Token Error Rate (TER)    : "
        f"{token_error_rate * 100:.2f}%"
    )

    print(
        f"Token Accuracy            : "
        f"{token_accuracy * 100:.2f}%"
    )

    print(
        f"Average Edit Distance     : "
        f"{avg_edit_distance:.4f}"
    )

    print(
        f"Average Inference Time    : "
        f"{avg_time:.4f} sec/image"
    )

    print("=" * 70)


    # ==================================================
    # Save Results
    # ==================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR /
        "detailed_evaluation_results.csv"
    )

    pd.DataFrame(results).to_csv(
        output_file,
        index=False
    )

    print("\nDetailed results saved to:")
    print(output_file)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    evaluate()