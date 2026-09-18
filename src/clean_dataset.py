"""
clean_dataset.py

HME100K Dataset Cleaning and Leakage Control
--------------------------------------------

Purpose:
1. Preserve the original HME100K dataset.
2. Detect invalid/missing image-label records.
3. Detect corrupted images.
4. Detect duplicate images within train/test.
5. Detect exact image overlap between train and test.
6. Remove exact duplicate samples from a CLEANED copy.
7. Preserve token-level mathematical labels.
8. Generate detailed audit logs.
9. Generate before-vs-after statistics.

IMPORTANT:
- Original HME100K is NEVER modified.
- Test labels are NEVER used to build vocabulary.
- Mathematical-expression overlap is NOT treated as image leakage.
"""

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

from src.config import (
    DATASET_DIR,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_FILE,
    TEST_IMAGE_DIR,
    TEST_LABEL_FILE,
)


# ============================================================
# Configuration
# ============================================================

CLEANED_DATASET_DIR = DATASET_DIR.parent / "HME100K_CLEANED"

CLEANED_TRAIN_DIR = (
    CLEANED_DATASET_DIR / "train" / "train_images"
)

CLEANED_TEST_DIR = (
    CLEANED_DATASET_DIR / "test" / "test_images"
)

CLEANED_TRAIN_LABEL_FILE = (
    CLEANED_DATASET_DIR / "train" / "train_labels.txt"
)

CLEANED_TEST_LABEL_FILE = (
    CLEANED_DATASET_DIR / "test" / "test_labels.txt"
)

AUDIT_DIR = CLEANED_DATASET_DIR / "audit"

REMOVAL_LOG = AUDIT_DIR / "removal_log.csv"
CONFLICT_LOG = AUDIT_DIR / "label_conflicts.csv"
DUPLICATE_LOG = AUDIT_DIR / "duplicate_groups.csv"
CROSS_SPLIT_LOG = AUDIT_DIR / "train_test_overlap.csv"
SUMMARY_JSON = AUDIT_DIR / "cleaning_summary.json"
SUMMARY_TXT = AUDIT_DIR / "cleaning_summary.txt"


# ============================================================
# Utility functions
# ============================================================

def image_hash(image_path):
    """
    Calculate SHA-256 hash of the actual decoded image pixels.

    Using decoded pixels rather than only the raw file bytes
    allows detection of identical images saved using different
    file encodings or metadata.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        return None

    # Include shape and dtype so different image structures
    # do not accidentally produce equivalent byte streams.
    metadata = (
        str(image.shape).encode("utf-8")
        + str(image.dtype).encode("utf-8")
    )

    return hashlib.sha256(
        metadata + image.tobytes()
    ).hexdigest()


def read_label_file(label_file):
    """
    Read HME100K label file.

    Expected format:

        image_filename<TAB>token token token ...

    Returns:
        records = [
            {
                "image": filename,
                "label": label,
                "line_number": line number
            }
        ]

    Also records malformed lines separately.
    """

    records = []
    malformed = []

    with open(
        label_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, raw_line in enumerate(f, start=1):

            line = raw_line.rstrip("\n\r")

            if not line.strip():
                malformed.append({
                    "line_number": line_number,
                    "reason": "empty_line",
                    "content": line
                })
                continue

            parts = line.split("\t", maxsplit=1)

            if len(parts) != 2:

                malformed.append({
                    "line_number": line_number,
                    "reason": "missing_tab_separator",
                    "content": line
                })
                continue

            image_name = parts[0].strip()
            label = parts[1].strip()

            if not image_name:

                malformed.append({
                    "line_number": line_number,
                    "reason": "empty_image_name",
                    "content": line
                })
                continue

            if not label:

                malformed.append({
                    "line_number": line_number,
                    "reason": "empty_label",
                    "content": line
                })
                continue

            # Verify that token-level representation exists.
            tokens = label.split()

            if len(tokens) == 0:

                malformed.append({
                    "line_number": line_number,
                    "reason": "zero_tokens",
                    "content": line
                })
                continue

            records.append({
                "image": image_name,
                "label": label,
                "line_number": line_number
            })

    return records, malformed


def get_image_files(image_dir):
    """
    Return image files recursively.

    Supported extensions are common image formats used by HME100K.
    """

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
    }

    return {
        p.name: p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
    }


def write_csv(path, rows, fieldnames):
    """
    Write rows to CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def safe_copy(source, destination):
    """
    Copy one image while preserving metadata where possible.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )


# ============================================================
# Audit one partition
# ============================================================

def audit_partition(
    partition_name,
    image_dir,
    label_file
):

    print("\n" + "=" * 70)
    print(f"AUDITING {partition_name.upper()} DATA")
    print("=" * 70)

    records, malformed = read_label_file(
        label_file
    )

    image_files = get_image_files(
        image_dir
    )

    label_names = {
        record["image"]
        for record in records
    }

    image_names = set(
        image_files.keys()
    )

    # --------------------------------------------------------
    # Missing records
    # --------------------------------------------------------

    missing_images = sorted(
        label_names - image_names
    )

    images_without_labels = sorted(
        image_names - label_names
    )

    # --------------------------------------------------------
    # Duplicate filenames in label file
    # --------------------------------------------------------

    filename_counts = defaultdict(list)

    for record in records:

        filename_counts[
            record["image"]
        ].append(record)

    duplicate_label_filenames = {
        name: rows
        for name, rows in filename_counts.items()
        if len(rows) > 1
    }

    # --------------------------------------------------------
    # Image validation + hash
    # --------------------------------------------------------

    corrupted_images = []

    hashes = {}

    dimensions = defaultdict(int)

    modes = defaultdict(int)

    formats = defaultdict(int)

    for filename, path in image_files.items():

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:

            corrupted_images.append(
                filename
            )

            continue

        # Dimensions
        height, width = image.shape[:2]

        dimensions[
            f"{width}x{height}"
        ] += 1

        # Mode/channel information
        if len(image.shape) == 2:

            modes["L"] += 1

        elif image.shape[2] == 3:

            modes["RGB/BGR"] += 1

        elif image.shape[2] == 4:

            modes["RGBA/BGRA"] += 1

        else:

            modes[
                f"{image.shape[2]} channels"
            ] += 1

        # File format
        formats[
            path.suffix.lower()
        ] += 1

        # Hash
        h = image_hash(path)

        if h is not None:

            hashes[filename] = h

    # --------------------------------------------------------
    # Internal duplicate groups
    # --------------------------------------------------------

    hash_groups = defaultdict(list)

    for filename, h in hashes.items():

        hash_groups[h].append(
            filename
        )

    duplicate_groups = {
        h: files
        for h, files in hash_groups.items()
        if len(files) > 1
    }

    print(
        f"Label records          : {len(records)}"
    )

    print(
        f"Image files            : {len(image_files)}"
    )

    print(
        f"Malformed records      : {len(malformed)}"
    )

    print(
        f"Missing images         : {len(missing_images)}"
    )

    print(
        f"Images without labels  : "
        f"{len(images_without_labels)}"
    )

    print(
        f"Corrupted images       : "
        f"{len(corrupted_images)}"
    )

    print(
        f"Duplicate label names  : "
        f"{len(duplicate_label_filenames)}"
    )

    print(
        f"Duplicate image groups : "
        f"{len(duplicate_groups)}"
    )

    return {
        "partition": partition_name,
        "records": records,
        "malformed": malformed,
        "image_files": image_files,
        "missing_images": missing_images,
        "images_without_labels": images_without_labels,
        "corrupted_images": corrupted_images,
        "duplicate_label_filenames": duplicate_label_filenames,
        "hashes": hashes,
        "duplicate_groups": duplicate_groups,
        "dimensions": dict(dimensions),
        "modes": dict(modes),
        "formats": dict(formats),
    }


# ============================================================
# Cross-partition exact image overlap
# ============================================================

def find_cross_split_overlaps(
    train_audit,
    test_audit
):

    print("\n" + "=" * 70)
    print("TRAIN-TEST EXACT IMAGE OVERLAP")
    print("=" * 70)

    train_hash_to_files = defaultdict(list)
    test_hash_to_files = defaultdict(list)

    for filename, h in train_audit["hashes"].items():

        train_hash_to_files[h].append(
            filename
        )

    for filename, h in test_audit["hashes"].items():

        test_hash_to_files[h].append(
            filename
        )

    common_hashes = sorted(
        set(train_hash_to_files)
        &
        set(test_hash_to_files)
    )

    overlaps = []

    for h in common_hashes:

        for train_file in train_hash_to_files[h]:

            for test_file in test_hash_to_files[h]:

                overlaps.append({
                    "hash": h,
                    "train_image": train_file,
                    "test_image": test_file
                })

    print(
        f"Exact train-test image overlaps: "
        f"{len(overlaps)}"
    )

    return overlaps


# ============================================================
# Compare labels
# ============================================================

def build_label_map(records):

    result = defaultdict(list)

    for record in records:

        result[
            record["image"]
        ].append(
            record["label"]
        )

    return result


def analyse_overlap_labels(
    overlaps,
    train_records,
    test_records
):

    train_labels = build_label_map(
        train_records
    )

    test_labels = build_label_map(
        test_records
    )

    conflicts = []

    for overlap in overlaps:

        train_file = overlap[
            "train_image"
        ]

        test_file = overlap[
            "test_image"
        ]

        train_set = set(
            train_labels.get(
                train_file,
                []
            )
        )

        test_set = set(
            test_labels.get(
                test_file,
                []
            )
        )

        if train_set != test_set:

            conflicts.append({
                "hash": overlap["hash"],
                "train_image": train_file,
                "test_image": test_file,
                "train_labels": " || ".join(
                    train_set
                ),
                "test_labels": " || ".join(
                    test_set
                )
            })

    return conflicts


# ============================================================
# Cleaning decision
# ============================================================

def build_removal_plan(
    train_audit,
    test_audit,
    cross_overlaps
):

    removals = []

    # --------------------------------------------------------
    # 1. Remove invalid train records
    # --------------------------------------------------------

    for filename in train_audit[
        "missing_images"
    ]:

        removals.append({
            "partition": "train",
            "image": filename,
            "reason": "missing_image"
        })

    for filename in train_audit[
        "corrupted_images"
    ]:

        removals.append({
            "partition": "train",
            "image": filename,
            "reason": "corrupted_image"
        })

    # --------------------------------------------------------
    # 2. Remove invalid test records
    # --------------------------------------------------------

    for filename in test_audit[
        "missing_images"
    ]:

        removals.append({
            "partition": "test",
            "image": filename,
            "reason": "missing_image"
        })

    for filename in test_audit[
        "corrupted_images"
    ]:

        removals.append({
            "partition": "test",
            "image": filename,
            "reason": "corrupted_image"
        })

    # --------------------------------------------------------
    # 3. Internal train duplicate groups
    #
    # Retain the first filename.
    # Remove additional identical images.
    # --------------------------------------------------------

    for h, files in train_audit[
        "duplicate_groups"
    ].items():

        files = sorted(files)

        retained = files[0]

        for duplicate in files[1:]:

            removals.append({
                "partition": "train",
                "image": duplicate,
                "reason": (
                    "internal_exact_duplicate"
                ),
                "duplicate_of": retained,
                "hash": h
            })

    # --------------------------------------------------------
    # 4. Internal test duplicate groups
    # --------------------------------------------------------

    for h, files in test_audit[
        "duplicate_groups"
    ].items():

        files = sorted(files)

        retained = files[0]

        for duplicate in files[1:]:

            removals.append({
                "partition": "test",
                "image": duplicate,
                "reason": (
                    "internal_exact_duplicate"
                ),
                "duplicate_of": retained,
                "hash": h
            })

    # --------------------------------------------------------
    # 5. Train-test exact image overlap
    #
    # Keep TEST copy.
    # Remove TRAIN copy.
    #
    # This prevents the same image from appearing in
    # both training and testing.
    # --------------------------------------------------------

    already_removed_train = {
        row["image"]
        for row in removals
        if row["partition"] == "train"
    }

    for overlap in cross_overlaps:

        train_file = overlap[
            "train_image"
        ]

        if train_file in already_removed_train:
            continue

        removals.append({
            "partition": "train",
            "image": train_file,
            "reason": (
                "exact_train_test_image_overlap"
            ),
            "test_duplicate": overlap[
                "test_image"
            ],
            "hash": overlap["hash"]
        })

        already_removed_train.add(
            train_file
        )

    return removals


# ============================================================
# Create cleaned dataset
# ============================================================

def create_cleaned_dataset(
    train_audit,
    test_audit,
    removals
):

    print("\n" + "=" * 70)
    print("CREATING CLEANED DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Safety: never delete original data
    # --------------------------------------------------------

    if CLEANED_DATASET_DIR.exists():

        print(
            f"Cleaning output already exists:\n"
            f"{CLEANED_DATASET_DIR}"
        )

        response = input(
            "\nDelete existing cleaned dataset "
            "and recreate it? [y/N]: "
        ).strip().lower()

        if response != "y":

            print(
                "Operation cancelled."
            )

            return False

        shutil.rmtree(
            CLEANED_DATASET_DIR
        )

    CLEANED_TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CLEANED_TEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Build removal sets
    # --------------------------------------------------------

    train_remove = {
        row["image"]
        for row in removals
        if row["partition"] == "train"
    }

    test_remove = {
        row["image"]
        for row in removals
        if row["partition"] == "test"
    }

    # --------------------------------------------------------
    # Copy TRAIN images
    # --------------------------------------------------------

    kept_train_records = []

    for record in train_audit["records"]:

        filename = record["image"]

        if filename in train_remove:
            continue

        source = (
            TRAIN_IMAGE_DIR /
            filename
        )

        destination = (
            CLEANED_TRAIN_DIR /
            filename
        )

        if source.exists():

            safe_copy(
                source,
                destination
            )

            kept_train_records.append(
                record
            )

    # --------------------------------------------------------
    # Copy TEST images
    # --------------------------------------------------------

    kept_test_records = []

    for record in test_audit["records"]:

        filename = record["image"]

        if filename in test_remove:
            continue

        source = (
            TEST_IMAGE_DIR /
            filename
        )

        destination = (
            CLEANED_TEST_DIR /
            filename
        )

        if source.exists():

            safe_copy(
                source,
                destination
            )

            kept_test_records.append(
                record
            )

    # --------------------------------------------------------
    # Write cleaned train labels
    # --------------------------------------------------------

    with open(
        CLEANED_TRAIN_LABEL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for record in kept_train_records:

            f.write(
                f"{record['image']}\t"
                f"{record['label']}\n"
            )

    # --------------------------------------------------------
    # Write cleaned test labels
    # --------------------------------------------------------

    with open(
        CLEANED_TEST_LABEL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for record in kept_test_records:

            f.write(
                f"{record['image']}\t"
                f"{record['label']}\n"
            )

    print(
        f"Cleaned training samples: "
        f"{len(kept_train_records)}"
    )

    print(
        f"Cleaned testing samples: "
        f"{len(kept_test_records)}"
    )

    return True


# ============================================================
# Write audit logs
# ============================================================

def write_logs(
    train_audit,
    test_audit,
    overlaps,
    conflicts,
    removals
):

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Removal log
    # --------------------------------------------------------

    removal_fields = [
        "partition",
        "image",
        "reason",
        "duplicate_of",
        "test_duplicate",
        "hash"
    ]

    write_csv(
        REMOVAL_LOG,
        removals,
        removal_fields
    )

    # --------------------------------------------------------
    # Duplicate group log
    # --------------------------------------------------------

    duplicate_rows = []

    for partition, audit in [
        ("train", train_audit),
        ("test", test_audit)
    ]:

        for h, files in audit[
            "duplicate_groups"
        ].items():

            for filename in sorted(files):

                duplicate_rows.append({
                    "partition": partition,
                    "hash": h,
                    "group_size": len(files),
                    "image": filename
                })

    write_csv(
        DUPLICATE_LOG,
        duplicate_rows,
        [
            "partition",
            "hash",
            "group_size",
            "image"
        ]
    )

    # --------------------------------------------------------
    # Cross split overlap log
    # --------------------------------------------------------

    write_csv(
        CROSS_SPLIT_LOG,
        overlaps,
        [
            "hash",
            "train_image",
            "test_image"
        ]
    )

    # --------------------------------------------------------
    # Label conflict log
    # --------------------------------------------------------

    write_csv(
        CONFLICT_LOG,
        conflicts,
        [
            "hash",
            "train_image",
            "test_image",
            "train_labels",
            "test_labels"
        ]
    )


# ============================================================
# Summary
# ============================================================

def create_summary(
    train_audit,
    test_audit,
    overlaps,
    conflicts,
    removals
):

    train_removed = {
        row["image"]
        for row in removals
        if row["partition"] == "train"
    }

    test_removed = {
        row["image"]
        for row in removals
        if row["partition"] == "test"
    }

    summary = {

        "dataset": "HME100K",

        "original": {

            "training_records":
                len(train_audit["records"]),

            "testing_records":
                len(test_audit["records"]),

            "training_images":
                len(train_audit["image_files"]),

            "testing_images":
                len(test_audit["image_files"]),

        },

        "quality_findings": {

            "training_malformed_records":
                len(train_audit["malformed"]),

            "testing_malformed_records":
                len(test_audit["malformed"]),

            "training_missing_images":
                len(train_audit["missing_images"]),

            "testing_missing_images":
                len(test_audit["missing_images"]),

            "training_images_without_labels":
                len(train_audit[
                    "images_without_labels"
                ]),

            "testing_images_without_labels":
                len(test_audit[
                    "images_without_labels"
                ]),

            "training_corrupted_images":
                len(train_audit[
                    "corrupted_images"
                ]),

            "testing_corrupted_images":
                len(test_audit[
                    "corrupted_images"
                ]),

        },

        "duplicate_analysis": {

            "training_duplicate_groups":
                len(train_audit[
                    "duplicate_groups"
                ]),

            "testing_duplicate_groups":
                len(test_audit[
                    "duplicate_groups"
                ]),

        },

        "leakage_analysis": {

            "exact_train_test_overlaps":
                len(overlaps),

            "label_conflicts_in_overlaps":
                len(conflicts),

        },

        "cleaning": {

            "training_removed":
                len(train_removed),

            "testing_removed":
                len(test_removed),

            "training_remaining":
                len(train_audit["records"])
                - len(train_removed),

            "testing_remaining":
                len(test_audit["records"])
                - len(test_removed),

        },

        "output_directory":
            str(CLEANED_DATASET_DIR),

        "vocabulary_policy":
            "Vocabulary must be rebuilt from cleaned "
            "training labels only.",

        "expression_overlap_policy":
            "Same mathematical expression in different "
            "handwritten images is not removed solely "
            "because the expression is shared.",

    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Human-readable TXT
    # --------------------------------------------------------

    with open(
        SUMMARY_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "HME100K DATASET CLEANING SUMMARY\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "ORIGINAL DATASET\n"
        )

        f.write(
            f"Training records : "
            f"{summary['original']['training_records']}\n"
        )

        f.write(
            f"Testing records  : "
            f"{summary['original']['testing_records']}\n"
        )

        f.write(
            f"Training images  : "
            f"{summary['original']['training_images']}\n"
        )

        f.write(
            f"Testing images   : "
            f"{summary['original']['testing_images']}\n\n"
        )

        f.write(
            "DATA QUALITY\n"
        )

        f.write(
            f"Train malformed : "
            f"{summary['quality_findings']['training_malformed_records']}\n"
        )

        f.write(
            f"Test malformed  : "
            f"{summary['quality_findings']['testing_malformed_records']}\n"
        )

        f.write(
            f"Train corrupted : "
            f"{summary['quality_findings']['training_corrupted_images']}\n"
        )

        f.write(
            f"Test corrupted  : "
            f"{summary['quality_findings']['testing_corrupted_images']}\n\n"
        )

        f.write(
            "DUPLICATES\n"
        )

        f.write(
            f"Train duplicate groups : "
            f"{summary['duplicate_analysis']['training_duplicate_groups']}\n"
        )

        f.write(
            f"Test duplicate groups  : "
            f"{summary['duplicate_analysis']['testing_duplicate_groups']}\n\n"
        )

        f.write(
            "LEAKAGE\n"
        )

        f.write(
            f"Exact train-test overlaps : "
            f"{summary['leakage_analysis']['exact_train_test_overlaps']}\n"
        )

        f.write(
            f"Label conflicts           : "
            f"{summary['leakage_analysis']['label_conflicts_in_overlaps']}\n\n"
        )

        f.write(
            "CLEANING\n"
        )

        f.write(
            f"Train removed : "
            f"{summary['cleaning']['training_removed']}\n"
        )

        f.write(
            f"Test removed  : "
            f"{summary['cleaning']['testing_removed']}\n"
        )

        f.write(
            f"Train remaining : "
            f"{summary['cleaning']['training_remaining']}\n"
        )

        f.write(
            f"Test remaining  : "
            f"{summary['cleaning']['testing_remaining']}\n\n"
        )

        f.write(
            "IMPORTANT POLICY\n"
        )

        f.write(
            "Shared mathematical expressions are not "
            "removed merely because they occur in both "
            "train and test.\n"
        )

        f.write(
            "Vocabulary must be rebuilt from cleaned "
            "training labels only.\n"
        )

    print(
        f"\nAudit files saved to:\n"
        f"{AUDIT_DIR}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("HME100K DATASET CLEANING PIPELINE")
    print("=" * 70)

    print(
        f"\nOriginal dataset:\n"
        f"{DATASET_DIR}"
    )

    print(
        f"\nCleaned dataset will be created at:\n"
        f"{CLEANED_DATASET_DIR}"
    )

    # --------------------------------------------------------
    # Verify original dataset
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{DATASET_DIR}"
        )

    if not TRAIN_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Training image directory not found:\n"
            f"{TRAIN_IMAGE_DIR}"
        )

    if not TEST_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Testing image directory not found:\n"
            f"{TEST_IMAGE_DIR}"
        )

    if not TRAIN_LABEL_FILE.exists():

        raise FileNotFoundError(
            f"Training label file not found:\n"
            f"{TRAIN_LABEL_FILE}"
        )

    if not TEST_LABEL_FILE.exists():

        raise FileNotFoundError(
            f"Testing label file not found:\n"
            f"{TEST_LABEL_FILE}"
        )

    # --------------------------------------------------------
    # Audit train
    # --------------------------------------------------------

    train_audit = audit_partition(
        "train",
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_FILE
    )

    # --------------------------------------------------------
    # Audit test
    # --------------------------------------------------------

    test_audit = audit_partition(
        "test",
        TEST_IMAGE_DIR,
        TEST_LABEL_FILE
    )

    # --------------------------------------------------------
    # Exact train-test overlap
    # --------------------------------------------------------

    overlaps = find_cross_split_overlaps(
        train_audit,
        test_audit
    )

    # --------------------------------------------------------
    # Check whether overlapping images have conflicting labels
    # --------------------------------------------------------

    conflicts = analyse_overlap_labels(
        overlaps,
        train_audit["records"],
        test_audit["records"]
    )

    print(
        f"Overlaps with label conflicts: "
        f"{len(conflicts)}"
    )

    # --------------------------------------------------------
    # Build cleaning plan
    # --------------------------------------------------------

    removals = build_removal_plan(
        train_audit,
        test_audit,
        overlaps
    )

    print("\n" + "=" * 70)
    print("CLEANING PLAN")
    print("=" * 70)

    train_removals = [
        x for x in removals
        if x["partition"] == "train"
    ]

    test_removals = [
        x for x in removals
        if x["partition"] == "test"
    ]

    print(
        f"Training records to remove : "
        f"{len(train_removals)}"
    )

    print(
        f"Testing records to remove  : "
        f"{len(test_removals)}"
    )

    print(
        "\nOriginal dataset WILL NOT be modified."
    )

    # --------------------------------------------------------
    # Write preliminary logs
    # --------------------------------------------------------

    write_logs(
        train_audit,
        test_audit,
        overlaps,
        conflicts,
        removals
    )

    # --------------------------------------------------------
    # Ask before creating cleaned dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    response = input(
        "Create the cleaned dataset now? [y/N]: "
    ).strip().lower()

    if response != "y":

        print(
            "\nCleaning cancelled."
        )

        print(
            f"Audit logs have still been saved to:\n"
            f"{AUDIT_DIR}"
        )

        return

    # --------------------------------------------------------
    # Create cleaned dataset
    # --------------------------------------------------------

    success = create_cleaned_dataset(
        train_audit,
        test_audit,
        removals
    )

    if not success:
        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    create_summary(
        train_audit,
        test_audit,
        overlaps,
        conflicts,
        removals
    )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETED")
    print("=" * 70)

    print(
        f"\nCleaned dataset:\n"
        f"{CLEANED_DATASET_DIR}"
    )

    print(
        f"\nAudit directory:\n"
        f"{AUDIT_DIR}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT run training yet."
    )

    print(
        "First perform the POST-CLEANING AUDIT."
    )

    print(
        "Then rebuild the vocabulary from the "
        "cleaned TRAINING labels only."
    )


if __name__ == "__main__":
    main()