"""
HME100K Dataset Cleaning and Leakage Control
============================================

Purpose
-------
1. Preserve the original HME100K dataset.
2. Audit missing/malformed image-label records.
3. Detect corrupted images.
4. Detect duplicate images within train/test.
5. Detect exact image-content overlap between train and test.
6. Distinguish unique overlap groups from overlap pairings.
7. Detect label conflicts in exact image overlaps.
8. Create a separate cleaned dataset.
9. Preserve token-level mathematical labels.
10. Generate detailed audit logs.
11. Generate before-vs-cleaning statistics.

IMPORTANT
---------
- Original HME100K is NEVER modified.
- Test labels are NEVER used to build vocabulary.
- Same mathematical expression in different handwritten images
  is NOT treated as image leakage.
- Exact image-content overlap between train and test is treated
  as a data-separation problem.
- Exact-overlap label conflicts are NOT silently resolved.
- Conflicting labels are preserved in the source partitions,
  while the TRAIN copy of an exact train-test duplicate is removed.
- The TEST copy is retained.
- Vocabulary must be rebuilt later from CLEANED TRAIN labels only.
"""

# ============================================================
# Imports
# ============================================================

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

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

# Original dataset
#
#     /content/HME100K
#
# Cleaned dataset
#
#     /content/HME100K_CLEANED
#
# The original dataset is NEVER modified.

CLEANED_DATASET_DIR = (
    DATASET_DIR.parent / "HME100K_CLEANED"
)

CLEANED_TRAIN_DIR = (
    CLEANED_DATASET_DIR
    / "train"
    / "train_images"
)

CLEANED_TEST_DIR = (
    CLEANED_DATASET_DIR
    / "test"
    / "test_images"
)

CLEANED_TRAIN_LABEL_FILE = (
    CLEANED_DATASET_DIR
    / "train"
    / "train_labels.txt"
)

CLEANED_TEST_LABEL_FILE = (
    CLEANED_DATASET_DIR
    / "test"
    / "test_labels.txt"
)


# ============================================================
# Audit output directory
# ============================================================

AUDIT_DIR = (
    CLEANED_DATASET_DIR / "audit"
)

REMOVAL_LOG = (
    AUDIT_DIR / "removal_log.csv"
)

CONFLICT_LOG = (
    AUDIT_DIR / "label_conflicts.csv"
)

DUPLICATE_LOG = (
    AUDIT_DIR / "duplicate_groups.csv"
)

CROSS_SPLIT_LOG = (
    AUDIT_DIR / "train_test_overlap.csv"
)

SUMMARY_JSON = (
    AUDIT_DIR / "cleaning_summary.json"
)

SUMMARY_TXT = (
    AUDIT_DIR / "cleaning_summary.txt"
)

MALFORMED_LOG = (
    AUDIT_DIR / "malformed_records.csv"
)

MISSING_LOG = (
    AUDIT_DIR / "missing_records.csv"
)


# ============================================================
# Utility Functions
# ============================================================

def image_hash(image_path):
    """
    Calculate SHA-256 hash of decoded image content.

    The hash includes:
        - image shape
        - image dtype
        - decoded pixel bytes

    This allows exact image-content duplication to be detected
    even if file metadata or filename differs.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        return None

    metadata = (
        str(image.shape).encode("utf-8")
        +
        str(image.dtype).encode("utf-8")
    )

    return hashlib.sha256(
        metadata + image.tobytes()
    ).hexdigest()


# ============================================================
# Read Label File
# ============================================================

def read_label_file(label_file):
    """
    Read an HME100K label file.

    Expected format:

        image_filename<TAB>token token token ...

    Example:

        000001.png    1 0 \\times ( x + 5 )

    Returns:
        records
        malformed records
    """

    records = []
    malformed = []

    with open(
        label_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, raw_line in enumerate(
            f,
            start=1
        ):

            line = raw_line.rstrip("\n\r")

            # ------------------------------------------------
            # Empty line
            # ------------------------------------------------

            if not line.strip():

                malformed.append({
                    "line_number": line_number,
                    "reason": "empty_line",
                    "content": line
                })

                continue

            # ------------------------------------------------
            # Image + label separation
            # ------------------------------------------------

            parts = line.split(
                "\t",
                maxsplit=1
            )

            if len(parts) != 2:

                malformed.append({
                    "line_number": line_number,
                    "reason": "missing_tab_separator",
                    "content": line
                })

                continue

            image_name = parts[0].strip()
            label = parts[1].strip()

            # ------------------------------------------------
            # Empty image name
            # ------------------------------------------------

            if not image_name:

                malformed.append({
                    "line_number": line_number,
                    "reason": "empty_image_name",
                    "content": line
                })

                continue

            # ------------------------------------------------
            # Empty label
            # ------------------------------------------------

            if not label:

                malformed.append({
                    "line_number": line_number,
                    "reason": "empty_label",
                    "content": line
                })

                continue

            # ------------------------------------------------
            # Token validation
            # ------------------------------------------------

            tokens = label.split()

            if len(tokens) == 0:

                malformed.append({
                    "line_number": line_number,
                    "reason": "zero_tokens",
                    "content": line
                })

                continue

            # ------------------------------------------------
            # Valid record
            # ------------------------------------------------

            records.append({
                "image": image_name,
                "label": label,
                "line_number": line_number
            })

    return records, malformed


# ============================================================
# Get Image Files
# ============================================================

def get_image_files(image_dir):
    """
    Return all supported image files directly inside the
    specified image directory.
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

    if not image_dir.exists():
        return {}

    return {
        path.name: path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    }


# ============================================================
# CSV Writer
# ============================================================

def write_csv(
    path,
    rows,
    fieldnames
):
    """
    Write rows to a CSV file.
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


# ============================================================
# Safe Image Copy
# ============================================================

def safe_copy(
    source,
    destination
):
    """
    Copy image without modifying the original.
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
# Audit One Partition
# ============================================================

def audit_partition(
    partition_name,
    image_dir,
    label_file
):

    print("\n" + "=" * 70)
    print(
        f"AUDITING {partition_name.upper()} DATA"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    records, malformed = (
        read_label_file(
            label_file
        )
    )

    # --------------------------------------------------------
    # Read images
    # --------------------------------------------------------

    image_files = (
        get_image_files(
            image_dir
        )
    )

    label_names = {
        record["image"]
        for record in records
    }

    image_names = set(
        image_files.keys()
    )

    # --------------------------------------------------------
    # Missing image records
    # --------------------------------------------------------

    missing_images = sorted(
        label_names - image_names
    )

    # --------------------------------------------------------
    # Images without labels
    # --------------------------------------------------------

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
        for name, rows
        in filename_counts.items()
        if len(rows) > 1
    }

    # --------------------------------------------------------
    # Image validation
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

        # ----------------------------------------------------
        # Corrupted/unreadable image
        # ----------------------------------------------------

        if image is None:

            corrupted_images.append(
                filename
            )

            continue

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        height, width = (
            image.shape[:2]
        )

        dimensions[
            f"{width}x{height}"
        ] += 1

        # ----------------------------------------------------
        # Image mode/channels
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # File format
        # ----------------------------------------------------

        formats[
            path.suffix.lower()
        ] += 1

        # ----------------------------------------------------
        # Image hash
        # ----------------------------------------------------

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
        h: sorted(files)
        for h, files
        in hash_groups.items()
        if len(files) > 1
    }

    # --------------------------------------------------------
    # Duplicate statistics
    # --------------------------------------------------------

    duplicate_images_involved = sum(
        len(files)
        for files
        in duplicate_groups.values()
    )

    redundant_duplicate_images = sum(
        len(files) - 1
        for files
        in duplicate_groups.values()
    )

    # --------------------------------------------------------
    # Print audit
    # --------------------------------------------------------

    print(
        f"Label records              : "
        f"{len(records)}"
    )

    print(
        f"Image files                : "
        f"{len(image_files)}"
    )

    print(
        f"Malformed records          : "
        f"{len(malformed)}"
    )

    print(
        f"Missing images             : "
        f"{len(missing_images)}"
    )

    print(
        f"Images without labels      : "
        f"{len(images_without_labels)}"
    )

    print(
        f"Corrupted images           : "
        f"{len(corrupted_images)}"
    )

    print(
        f"Duplicate label names      : "
        f"{len(duplicate_label_filenames)}"
    )

    print(
        f"Duplicate image groups     : "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Duplicate images involved  : "
        f"{duplicate_images_involved}"
    )

    print(
        f"Redundant duplicate images : "
        f"{redundant_duplicate_images}"
    )

    return {

        "partition":
            partition_name,

        "records":
            records,

        "malformed":
            malformed,

        "image_files":
            image_files,

        "missing_images":
            missing_images,

        "images_without_labels":
            images_without_labels,

        "corrupted_images":
            corrupted_images,

        "duplicate_label_filenames":
            duplicate_label_filenames,

        "hashes":
            hashes,

        "duplicate_groups":
            duplicate_groups,

        "duplicate_images_involved":
            duplicate_images_involved,

        "redundant_duplicate_images":
            redundant_duplicate_images,

        "dimensions":
            dict(dimensions),

        "modes":
            dict(modes),

        "formats":
            dict(formats),
    }


# ============================================================
# Find Exact Train-Test Overlap
# ============================================================

def find_cross_split_overlaps(
    train_audit,
    test_audit
):
    """
    Detect exact image-content overlap.

    IMPORTANT:

    A single image-content hash may correspond to:

        Train:
            image_A
            image_B

        Test:
            image_C
            image_D

    This represents:

        1 unique overlap group/hash

    but:

        2 train images
        2 test images
        4 possible pairings

    Therefore these quantities are reported separately.
    """

    print("\n" + "=" * 70)
    print(
        "TRAIN-TEST EXACT IMAGE OVERLAP"
    )
    print("=" * 70)

    train_hash_to_files = defaultdict(list)
    test_hash_to_files = defaultdict(list)

    # --------------------------------------------------------
    # Train hash groups
    # --------------------------------------------------------

    for filename, h in (
        train_audit["hashes"].items()
    ):

        train_hash_to_files[h].append(
            filename
        )

    # --------------------------------------------------------
    # Test hash groups
    # --------------------------------------------------------

    for filename, h in (
        test_audit["hashes"].items()
    ):

        test_hash_to_files[h].append(
            filename
        )

    # --------------------------------------------------------
    # Common hashes
    # --------------------------------------------------------

    common_hashes = sorted(
        set(train_hash_to_files)
        &
        set(test_hash_to_files)
    )

    overlap_groups = []

    # --------------------------------------------------------
    # Build unique overlap groups
    # --------------------------------------------------------

    for h in common_hashes:

        train_files = sorted(
            train_hash_to_files[h]
        )

        test_files = sorted(
            test_hash_to_files[h]
        )

        pair_count = (
            len(train_files)
            *
            len(test_files)
        )

        overlap_groups.append({

            "hash":
                h,

            "train_images":
                train_files,

            "test_images":
                test_files,

            "train_count":
                len(train_files),

            "test_count":
                len(test_files),

            "pair_count":
                pair_count,
        })

    # --------------------------------------------------------
    # Unique image counts
    # --------------------------------------------------------

    unique_train_images = {
        filename
        for group
        in overlap_groups
        for filename
        in group["train_images"]
    }

    unique_test_images = {
        filename
        for group
        in overlap_groups
        for filename
        in group["test_images"]
    }

    # --------------------------------------------------------
    # Pair count
    # --------------------------------------------------------

    total_pairings = sum(
        group["pair_count"]
        for group
        in overlap_groups
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Unique overlapping hashes       : "
        f"{len(overlap_groups)}"
    )

    print(
        f"Unique training images involved  : "
        f"{len(unique_train_images)}"
    )

    print(
        f"Unique testing images involved   : "
        f"{len(unique_test_images)}"
    )

    print(
        f"Total train-test overlap pairs   : "
        f"{total_pairings}"
    )

    return overlap_groups


# ============================================================
# Build Label Map
# ============================================================

def build_label_map(records):
    """
    Build:

        image filename
            ->
        list of associated labels
    """

    result = defaultdict(list)

    for record in records:

        result[
            record["image"]
        ].append(
            record["label"]
        )

    return result


# ============================================================
# Analyse Overlap Labels
# ============================================================

def analyse_overlap_labels(
    overlaps,
    train_records,
    test_records
):
    """
    Determine whether exact-overlap groups have different labels.

    Comparison is performed at the UNIQUE HASH/GROUP level.

    Duplicate train/test pairings therefore do not inflate the
    conflict count.
    """

    train_labels = (
        build_label_map(
            train_records
        )
    )

    test_labels = (
        build_label_map(
            test_records
        )
    )

    conflicts = []

    for group in overlaps:

        train_label_values = sorted({
            label
            for filename
            in group["train_images"]
            for label
            in train_labels.get(
                filename,
                []
            )
        })

        test_label_values = sorted({
            label
            for filename
            in group["test_images"]
            for label
            in test_labels.get(
                filename,
                []
            )
        })

        # ----------------------------------------------------
        # Label conflict
        # ----------------------------------------------------

        if (
            set(train_label_values)
            !=
            set(test_label_values)
        ):

            conflicts.append({

                "hash":
                    group["hash"],

                "train_images":
                    " || ".join(
                        group["train_images"]
                    ),

                "test_images":
                    " || ".join(
                        group["test_images"]
                    ),

                "train_labels":
                    " || ".join(
                        train_label_values
                    ),

                "test_labels":
                    " || ".join(
                        test_label_values
                    ),

                "train_count":
                    group["train_count"],

                "test_count":
                    group["test_count"],
            })

    print(
        f"Unique overlap groups with "
        f"label conflicts: "
        f"{len(conflicts)}"
    )

    return conflicts


# ============================================================
# Build Cleaning Plan
# ============================================================

def build_removal_plan(
    train_audit,
    test_audit,
    cross_overlaps
):
    """
    Build the cleaning/removal plan.

    Rules
    -----

    1. Missing images:
       Remove invalid records from the cleaned label set.

    2. Corrupted images:
       Remove.

    3. Images without labels:
       Do not copy them to the cleaned dataset.

    4. Internal exact duplicates:
       Retain the first sorted filename.
       Remove redundant copies.

    5. Exact train-test image overlap:
       Retain TEST copy.
       Remove TRAIN copies.

    6. Label conflicts:
       DO NOT choose or modify a label.
       The conflicting TEST label is preserved.
       The conflicting TRAIN image is removed because it is
       an exact duplicate of a TEST image.

    This prevents the same image content from appearing in
    both partitions.
    """

    removals = []

    # ========================================================
    # TRAIN INVALID / UNUSABLE RECORDS
    # ========================================================

    for filename in (
        train_audit["missing_images"]
    ):

        removals.append({

            "partition":
                "train",

            "image":
                filename,

            "reason":
                "missing_image",

            "duplicate_of":
                "",

            "test_duplicate":
                "",

            "hash":
                "",
        })

    for filename in (
        train_audit["corrupted_images"]
    ):

        removals.append({

            "partition":
                "train",

            "image":
                filename,

            "reason":
                "corrupted_image",

            "duplicate_of":
                "",

            "test_duplicate":
                "",

            "hash":
                "",
        })

    for filename in (
        train_audit["images_without_labels"]
    ):

        removals.append({

            "partition":
                "train",

            "image":
                filename,

            "reason":
                "image_without_label",

            "duplicate_of":
                "",

            "test_duplicate":
                "",

            "hash":
                train_audit["hashes"].get(
                    filename,
                    ""
                ),
        })

    # ========================================================
    # TEST INVALID / UNUSABLE RECORDS
    # ========================================================

    for filename in (
        test_audit["missing_images"]
    ):

        removals.append({

            "partition":
                "test",

            "image":
                filename,

            "reason":
                "missing_image",

            "duplicate_of":
                "",

            "test_duplicate":
                "",

            "hash":
                "",
        })

    for filename in (
        test_audit["corrupted_images"]
    ):

        removals.append({

            "partition":
                "test",

            "image":
                filename,

            "reason":
                "corrupted_image",

            "duplicate_of":
                "",

            "test_duplicate":
                "",

            "hash":
                "",
        })

    for filename in (
        test_audit["images_without_labels"]
    ):

        removals.append({

            "partition":
                "test",

            "image":
                filename,

            "reason":
                "image_without_label",

            "duplicate_of":
                "",

            "test_duplicate":
                "",

            "hash":
                test_audit["hashes"].get(
                    filename,
                    ""
                ),
        })

    # ========================================================
    # INTERNAL TRAIN DUPLICATES
    # ========================================================

    for h, files in (
        train_audit[
            "duplicate_groups"
        ].items()
    ):

        files = sorted(files)

        retained = files[0]

        for duplicate in files[1:]:

            removals.append({

                "partition":
                    "train",

                "image":
                    duplicate,

                "reason":
                    "internal_exact_duplicate",

                "duplicate_of":
                    retained,

                "test_duplicate":
                    "",

                "hash":
                    h,
            })

    # ========================================================
    # INTERNAL TEST DUPLICATES
    # ========================================================

    for h, files in (
        test_audit[
            "duplicate_groups"
        ].items()
    ):

        files = sorted(files)

        retained = files[0]

        for duplicate in files[1:]:

            removals.append({

                "partition":
                    "test",

                "image":
                    duplicate,

                "reason":
                    "internal_exact_duplicate",

                "duplicate_of":
                    retained,

                "test_duplicate":
                    "",

                "hash":
                    h,
            })

    # ========================================================
    # TRAIN-TEST EXACT IMAGE OVERLAP
    # ========================================================

    already_removed_train = {
        row["image"]
        for row in removals
        if row["partition"] == "train"
    }

    for group in cross_overlaps:

        for train_file in (
            group["train_images"]
        ):

            if train_file in (
                already_removed_train
            ):
                continue

            removals.append({

                "partition":
                    "train",

                "image":
                    train_file,

                "reason":
                    "exact_train_test_image_overlap",

                "duplicate_of":
                    "",

                "test_duplicate":
                    " || ".join(
                        group["test_images"]
                    ),

                "hash":
                    group["hash"],
            })

            already_removed_train.add(
                train_file
            )

    return removals


# ============================================================
# Create Cleaned Dataset
# ============================================================

def create_cleaned_dataset(
    train_audit,
    test_audit,
    removals
):
    """
    Create a separate cleaned dataset.

    Original HME100K is never modified.
    """

    print("\n" + "=" * 70)
    print(
        "CREATING CLEANED DATASET"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Existing cleaned directory
    # --------------------------------------------------------

    if CLEANED_DATASET_DIR.exists():

        print(
            f"\nCleaning output already exists:\n"
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

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

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
    # Removal sets
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
    # Copy training data
    # --------------------------------------------------------

    kept_train_records = []

    copied_train_images = set()

    for record in train_audit[
        "records"
    ]:

        filename = record["image"]

        if filename in train_remove:
            continue

        source = (
            TRAIN_IMAGE_DIR
            / filename
        )

        destination = (
            CLEANED_TRAIN_DIR
            / filename
        )

        if source.exists():

            safe_copy(
                source,
                destination
            )

            kept_train_records.append(
                record
            )

            copied_train_images.add(
                filename
            )

    # --------------------------------------------------------
    # Copy testing data
    # --------------------------------------------------------

    kept_test_records = []

    copied_test_images = set()

    for record in test_audit[
        "records"
    ]:

        filename = record["image"]

        if filename in test_remove:
            continue

        source = (
            TEST_IMAGE_DIR
            / filename
        )

        destination = (
            CLEANED_TEST_DIR
            / filename
        )

        if source.exists():

            safe_copy(
                source,
                destination
            )

            kept_test_records.append(
                record
            )

            copied_test_images.add(
                filename
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

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nCleaned training samples : "
        f"{len(kept_train_records)}"
    )

    print(
        f"Cleaned testing samples  : "
        f"{len(kept_test_records)}"
    )

    print(
        f"Training images copied   : "
        f"{len(copied_train_images)}"
    )

    print(
        f"Testing images copied    : "
        f"{len(copied_test_images)}"
    )

    return True


# ============================================================
# Write Audit Logs
# ============================================================

def write_logs(
    train_audit,
    test_audit,
    overlaps,
    conflicts,
    removals
):
    """
    Write all detailed audit information.
    """

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Removal log
    # ========================================================

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

    # ========================================================
    # Duplicate group log
    # ========================================================

    duplicate_rows = []

    for partition, audit in [
        ("train", train_audit),
        ("test", test_audit)
    ]:

        for h, files in (
            audit[
                "duplicate_groups"
            ].items()
        ):

            files = sorted(files)

            for filename in files:

                duplicate_rows.append({

                    "partition":
                        partition,

                    "hash":
                        h,

                    "group_size":
                        len(files),

                    "redundant_images":
                        len(files) - 1,

                    "retained_image":
                        files[0],

                    "image":
                        filename,
                })

    write_csv(
        DUPLICATE_LOG,
        duplicate_rows,
        [
            "partition",
            "hash",
            "group_size",
            "redundant_images",
            "retained_image",
            "image"
        ]
    )

    # ========================================================
    # Train-test overlap log
    # ========================================================

    overlap_rows = []

    for group in overlaps:

        overlap_rows.append({

            "hash":
                group["hash"],

            "train_images":
                " || ".join(
                    group["train_images"]
                ),

            "test_images":
                " || ".join(
                    group["test_images"]
                ),

            "train_count":
                group["train_count"],

            "test_count":
                group["test_count"],

            "pair_count":
                group["pair_count"],
        })

    write_csv(
        CROSS_SPLIT_LOG,
        overlap_rows,
        [
            "hash",
            "train_images",
            "test_images",
            "train_count",
            "test_count",
            "pair_count"
        ]
    )

    # ========================================================
    # Label conflict log
    # ========================================================

    write_csv(
        CONFLICT_LOG,
        conflicts,
        [
            "hash",
            "train_images",
            "test_images",
            "train_labels",
            "test_labels",
            "train_count",
            "test_count"
        ]
    )

    # ========================================================
    # Malformed records
    # ========================================================

    malformed_rows = []

    for partition, audit in [
        ("train", train_audit),
        ("test", test_audit)
    ]:

        for row in audit[
            "malformed"
        ]:

            malformed_rows.append({

                "partition":
                    partition,

                "line_number":
                    row["line_number"],

                "reason":
                    row["reason"],

                "content":
                    row["content"],
            })

    write_csv(
        MALFORMED_LOG,
        malformed_rows,
        [
            "partition",
            "line_number",
            "reason",
            "content"
        ]
    )

    # ========================================================
    # Missing records
    # ========================================================

    missing_rows = []

    for partition, audit in [
        ("train", train_audit),
        ("test", test_audit)
    ]:

        for filename in audit[
            "missing_images"
        ]:

            missing_rows.append({

                "partition":
                    partition,

                "image":
                    filename,

                "reason":
                    "label_references_missing_image"
            })

        for filename in audit[
            "images_without_labels"
        ]:

            missing_rows.append({

                "partition":
                    partition,

                "image":
                    filename,

                "reason":
                    "image_has_no_label"
            })

    write_csv(
        MISSING_LOG,
        missing_rows,
        [
            "partition",
            "image",
            "reason"
        ]
    )


# ============================================================
# Create Summary
# ============================================================

def create_summary(
    train_audit,
    test_audit,
    overlaps,
    conflicts,
    removals
):
    """
    Generate JSON and TXT summaries.
    """

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

    unique_overlap_train_images = {
        image
        for group in overlaps
        for image in group["train_images"]
    }

    unique_overlap_test_images = {
        image
        for group in overlaps
        for image in group["test_images"]
    }

    total_pairings = sum(
        group["pair_count"]
        for group in overlaps
    )

    # --------------------------------------------------------
    # Count removal reasons
    # --------------------------------------------------------

    removal_reason_counts = defaultdict(int)

    for row in removals:
        removal_reason_counts[
            row["reason"]
        ] += 1

    # --------------------------------------------------------
    # Summary dictionary
    # --------------------------------------------------------

    summary = {

        "dataset":
            "HME100K",

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
                len(
                    train_audit[
                        "images_without_labels"
                    ]
                ),

            "testing_images_without_labels":
                len(
                    test_audit[
                        "images_without_labels"
                    ]
                ),

            "training_corrupted_images":
                len(
                    train_audit[
                        "corrupted_images"
                    ]
                ),

            "testing_corrupted_images":
                len(
                    test_audit[
                        "corrupted_images"
                    ]
                ),

            "training_duplicate_label_names":
                len(
                    train_audit[
                        "duplicate_label_filenames"
                    ]
                ),

            "testing_duplicate_label_names":
                len(
                    test_audit[
                        "duplicate_label_filenames"
                    ]
                ),
        },

        "duplicate_analysis": {

            "training_duplicate_groups":
                len(
                    train_audit[
                        "duplicate_groups"
                    ]
                ),

            "testing_duplicate_groups":
                len(
                    test_audit[
                        "duplicate_groups"
                    ]
                ),

            "training_duplicate_images_involved":
                train_audit[
                    "duplicate_images_involved"
                ],

            "testing_duplicate_images_involved":
                test_audit[
                    "duplicate_images_involved"
                ],

            "training_redundant_duplicate_images":
                train_audit[
                    "redundant_duplicate_images"
                ],

            "testing_redundant_duplicate_images":
                test_audit[
                    "redundant_duplicate_images"
                ],
        },

        "leakage_analysis": {

            "unique_overlapping_image_hashes":
                len(overlaps),

            "unique_training_images_involved":
                len(
                    unique_overlap_train_images
                ),

            "unique_testing_images_involved":
                len(
                    unique_overlap_test_images
                ),

            "total_train_test_overlap_pairings":
                total_pairings,

            "unique_overlap_groups_with_label_conflicts":
                len(conflicts),
        },

        "cleaning": {

            "training_removed":
                len(train_removed),

            "testing_removed":
                len(test_removed),

            "training_remaining":
                len(
                    train_audit["records"]
                )
                -
                len(train_removed),

            "testing_remaining":
                len(
                    test_audit["records"]
                )
                -
                len(test_removed),

            "removal_reason_counts":
                dict(
                    removal_reason_counts
                ),
        },

        "output_directory":
            str(CLEANED_DATASET_DIR),

        "vocabulary_policy":
            "Vocabulary must be rebuilt from "
            "cleaned training labels only. "
            "Test labels are not used for vocabulary "
            "construction.",

        "expression_overlap_policy":
            "Same mathematical expression in different "
            "handwritten images is not removed solely "
            "because the expression is shared.",

        "label_conflict_policy":
            "Exact image-overlap label conflicts are "
            "logged and preserved for review. "
            "No label is automatically changed. "
            "The training copy is removed because the "
            "same image content exists in the test set.",

        "original_dataset_policy":
            "Original HME100K is never modified.",
    }

    # ========================================================
    # Save JSON
    # ========================================================

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

    # ========================================================
    # Save TXT
    # ========================================================

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

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Quality
        # ----------------------------------------------------

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
            f"{summary['quality_findings']['testing_corrupted_images']}\n"
        )

        f.write(
            f"Train missing images : "
            f"{summary['quality_findings']['training_missing_images']}\n"
        )

        f.write(
            f"Test missing images : "
            f"{summary['quality_findings']['testing_missing_images']}\n"
        )

        f.write(
            f"Train images without labels : "
            f"{summary['quality_findings']['training_images_without_labels']}\n"
        )

        f.write(
            f"Test images without labels : "
            f"{summary['quality_findings']['testing_images_without_labels']}\n\n"
        )

        # ----------------------------------------------------
        # Duplicates
        # ----------------------------------------------------

        f.write(
            "DUPLICATE ANALYSIS\n"
        )

        f.write(
            f"Train duplicate groups : "
            f"{summary['duplicate_analysis']['training_duplicate_groups']}\n"
        )

        f.write(
            f"Test duplicate groups  : "
            f"{summary['duplicate_analysis']['testing_duplicate_groups']}\n"
        )

        f.write(
            f"Train duplicate images involved : "
            f"{summary['duplicate_analysis']['training_duplicate_images_involved']}\n"
        )

        f.write(
            f"Test duplicate images involved : "
            f"{summary['duplicate_analysis']['testing_duplicate_images_involved']}\n"
        )

        f.write(
            f"Train redundant duplicates : "
            f"{summary['duplicate_analysis']['training_redundant_duplicate_images']}\n"
        )

        f.write(
            f"Test redundant duplicates : "
            f"{summary['duplicate_analysis']['testing_redundant_duplicate_images']}\n\n"
        )

        # ----------------------------------------------------
        # Leakage
        # ----------------------------------------------------

        f.write(
            "TRAIN-TEST EXACT IMAGE OVERLAP ANALYSIS\n"
        )

        f.write(
            f"Unique overlapping hashes : "
            f"{summary['leakage_analysis']['unique_overlapping_image_hashes']}\n"
        )

        f.write(
            f"Unique train images involved : "
            f"{summary['leakage_analysis']['unique_training_images_involved']}\n"
        )

        f.write(
            f"Unique test images involved : "
            f"{summary['leakage_analysis']['unique_testing_images_involved']}\n"
        )

        f.write(
            f"Total overlap pairings : "
            f"{summary['leakage_analysis']['total_train_test_overlap_pairings']}\n"
        )

        f.write(
            f"Overlap groups with label conflicts : "
            f"{summary['leakage_analysis']['unique_overlap_groups_with_label_conflicts']}\n\n"
        )

        # ----------------------------------------------------
        # Cleaning
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Removal reasons
        # ----------------------------------------------------

        f.write(
            "REMOVAL REASONS\n"
        )

        for reason, count in sorted(
            removal_reason_counts.items()
        ):

            f.write(
                f"{reason} : {count}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Policies
        # ----------------------------------------------------

        f.write(
            "POLICIES\n"
        )

        f.write(
            "Original HME100K is never modified.\n"
        )

        f.write(
            "Vocabulary is rebuilt from cleaned TRAIN "
            "labels only.\n"
        )

        f.write(
            "Shared mathematical expressions are not "
            "removed merely because they occur in both "
            "train and test.\n"
        )

        f.write(
            "Exact train-test image overlap is treated "
            "as a data-separation issue.\n"
        )

        f.write(
            "The TEST copy of an exact train-test duplicate "
            "is retained and the TRAIN copy is removed.\n"
        )

        f.write(
            "Label conflicts are logged and are not "
            "automatically corrected.\n"
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

    print(
        "HME100K DATASET CLEANING PIPELINE"
    )

    print("=" * 70)

    print(
        f"\nOriginal dataset:\n"
        f"{DATASET_DIR}"
    )

    print(
        f"\nCleaned dataset will be created at:\n"
        f"{CLEANED_DATASET_DIR}"
    )

    # ========================================================
    # Verify original dataset
    # ========================================================

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

    # ========================================================
    # Audit train
    # ========================================================

    train_audit = audit_partition(
        "train",
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_FILE
    )

    # ========================================================
    # Audit test
    # ========================================================

    test_audit = audit_partition(
        "test",
        TEST_IMAGE_DIR,
        TEST_LABEL_FILE
    )

    # ========================================================
    # Find exact train-test image overlaps
    # ========================================================

    overlaps = find_cross_split_overlaps(
        train_audit,
        test_audit
    )

    # ========================================================
    # Analyse labels
    # ========================================================

    conflicts = analyse_overlap_labels(
        overlaps,
        train_audit["records"],
        test_audit["records"]
    )

    # ========================================================
    # Build cleaning plan
    # ========================================================

    removals = build_removal_plan(
        train_audit,
        test_audit,
        overlaps
    )

    # ========================================================
    # Print cleaning plan
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "CLEANING PLAN"
    )

    print("=" * 70)

    train_removals = [
        row
        for row in removals
        if row["partition"] == "train"
    ]

    test_removals = [
        row
        for row in removals
        if row["partition"] == "test"
    ]

    # --------------------------------------------------------
    # Removal reason breakdown
    # --------------------------------------------------------

    train_reason_counts = defaultdict(int)
    test_reason_counts = defaultdict(int)

    for row in train_removals:
        train_reason_counts[
            row["reason"]
        ] += 1

    for row in test_removals:
        test_reason_counts[
            row["reason"]
        ] += 1

    print(
        f"\nUnique train records to remove : "
        f"{len({
            row['image']
            for row in train_removals
        })}"
    )

    print(
        f"Unique test records to remove  : "
        f"{len({
            row['image']
            for row in test_removals
        })}"
    )

    print("\nTRAIN REMOVAL BREAKDOWN")

    for reason, count in sorted(
        train_reason_counts.items()
    ):

        print(
            f"  {reason:<40} : {count}"
        )

    print("\nTEST REMOVAL BREAKDOWN")

    for reason, count in sorted(
        test_reason_counts.items()
    ):

        print(
            f"  {reason:<40} : {count}"
        )

    # --------------------------------------------------------
    # Leakage information
    # --------------------------------------------------------

    unique_overlap_train = {
        image
        for group in overlaps
        for image in group["train_images"]
    }

    unique_overlap_test = {
        image
        for group in overlaps
        for image in group["test_images"]
    }

    total_pairings = sum(
        group["pair_count"]
        for group in overlaps
    )

    print("\n" + "-" * 70)

    print(
        "EXACT TRAIN-TEST OVERLAP SUMMARY"
    )

    print("-" * 70)

    print(
        f"Unique overlap groups/hashes : "
        f"{len(overlaps)}"
    )

    print(
        f"Unique train images involved  : "
        f"{len(unique_overlap_train)}"
    )

    print(
        f"Unique test images involved   : "
        f"{len(unique_overlap_test)}"
    )

    print(
        f"Total overlap pairings        : "
        f"{total_pairings}"
    )

    print(
        f"Label-conflict groups         : "
        f"{len(conflicts)}"
    )

    print("\nIMPORTANT:")

    print(
        "The 23 label conflicts WILL NOT block cleaning."
    )

    print(
        "No label will be automatically changed."
    )

    print(
        "The TRAIN copy of exact train-test duplicates "
        "will be removed."
    )

    print(
        "The TEST copy will be retained."
    )

    print(
        "\nOriginal dataset WILL NOT be modified."
    )

    # ========================================================
    # Write audit logs BEFORE cleaning
    # ========================================================

    write_logs(
        train_audit,
        test_audit,
        overlaps,
        conflicts,
        removals
    )

    # ========================================================
    # Ask permission
    # ========================================================

    print("\n" + "=" * 70)

    response = input(
        "Create the cleaned dataset now? [y/N]: "
    ).strip().lower()

    if response != "y":

        print(
            "\nCleaning cancelled."
        )

        print(
            f"\nAudit logs have been saved to:\n"
            f"{AUDIT_DIR}"
        )

        return

    # ========================================================
    # Create cleaned dataset
    # ========================================================

    success = create_cleaned_dataset(
        train_audit,
        test_audit,
        removals
    )

    if not success:
        return

    # ========================================================
    # Create summary
    # ========================================================

    create_summary(
        train_audit,
        test_audit,
        overlaps,
        conflicts,
        removals
    )

    # ========================================================
    # Final message
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "CLEANING COMPLETED"
    )

    print("=" * 70)

    print(
        f"\nCleaned dataset:"
        f"\n{CLEANED_DATASET_DIR}"
    )

    print(
        f"\nAudit directory:"
        f"\n{AUDIT_DIR}"
    )

    print(
        "\nIMPORTANT NEXT STEP:"
    )

    print(
        "Do NOT start model training yet."
    )

    print(
        "First perform the POST-CLEANING AUDIT."
    )

    print(
        "Then rebuild the vocabulary from the "
        "cleaned TRAINING labels only."
    )

    print(
        "\nThe original HME100K dataset remains untouched."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()