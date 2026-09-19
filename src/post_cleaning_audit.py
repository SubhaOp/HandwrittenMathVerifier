"""
HME100K POST-CLEANING DATASET AUDIT
===================================

Purpose
-------
This script verifies the HME100K dataset AFTER cleaning.

IMPORTANT
---------
This script DOES NOT modify the cleaned dataset.

It only audits:

1. Image-label correspondence
2. Missing images
3. Images without labels
4. Malformed label records
5. Corrupted images
6. Duplicate label filenames
7. Internal exact-image duplicates
8. Exact train-test image overlap
9. Remaining label conflicts
10. Before/after dataset statistics
11. Cleaning effectiveness

Expected cleaned dataset:

    /content/HME100K_CLEANED/

Structure:

    HME100K_CLEANED/
    ├── train/
    │   ├── train_images/
    │   └── train_labels.txt
    │
    ├── test/
    │   ├── test_images/
    │   └── test_labels.txt
    │
    └── audit/
"""

# ============================================================
# IMPORTS
# ============================================================

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2


# ============================================================
# PATH CONFIGURATION
# ============================================================

CLEANED_DATASET_DIR = Path(
    "/content/HME100K_CLEANED"
)

TRAIN_IMAGE_DIR = (
    CLEANED_DATASET_DIR
    / "train"
    / "train_images"
)

TRAIN_LABEL_FILE = (
    CLEANED_DATASET_DIR
    / "train"
    / "train_labels.txt"
)

TEST_IMAGE_DIR = (
    CLEANED_DATASET_DIR
    / "test"
    / "test_images"
)

TEST_LABEL_FILE = (
    CLEANED_DATASET_DIR
    / "test"
    / "test_labels.txt"
)

AUDIT_DIR = (
    CLEANED_DATASET_DIR
    / "audit"
)


# ============================================================
# OUTPUT FILES
# ============================================================

POST_AUDIT_SUMMARY_JSON = (
    AUDIT_DIR
    / "post_cleaning_audit_summary.json"
)

POST_AUDIT_SUMMARY_TXT = (
    AUDIT_DIR
    / "post_cleaning_audit_summary.txt"
)

POST_DUPLICATES_CSV = (
    AUDIT_DIR
    / "post_cleaning_duplicate_groups.csv"
)

POST_OVERLAP_CSV = (
    AUDIT_DIR
    / "post_cleaning_train_test_overlap.csv"
)

POST_CONFLICTS_CSV = (
    AUDIT_DIR
    / "post_cleaning_label_conflicts.csv"
)

POST_MISSING_CSV = (
    AUDIT_DIR
    / "post_cleaning_missing_records.csv"
)

POST_MALFORMED_CSV = (
    AUDIT_DIR
    / "post_cleaning_malformed_records.csv"
)


# ============================================================
# EXPECTED PRE-CLEANING COUNTS
# ============================================================

# These are the verified values from the original dataset audit.

ORIGINAL_TRAIN_COUNT = 74502
ORIGINAL_TEST_COUNT = 24607

EXPECTED_TRAIN_REMOVED = 276
EXPECTED_TEST_REMOVED = 27

EXPECTED_TRAIN_COUNT = (
    ORIGINAL_TRAIN_COUNT
    - EXPECTED_TRAIN_REMOVED
)

EXPECTED_TEST_COUNT = (
    ORIGINAL_TEST_COUNT
    - EXPECTED_TEST_REMOVED
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_image_files(image_dir):
    """
    Return supported image files.
    """

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
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
# READ LABEL FILE
# ============================================================

def read_label_file(label_file):
    """
    Read label file.

    Expected format:

        image_filename<TAB>token token token ...

    Returns:

        records
        malformed_records
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

            line = raw_line.rstrip(
                "\n\r"
            )

            # ------------------------------------------------
            # Empty line
            # ------------------------------------------------

            if not line.strip():

                malformed.append({

                    "line_number":
                        line_number,

                    "reason":
                        "empty_line",

                    "content":
                        line,
                })

                continue

            # ------------------------------------------------
            # Split image and label
            # ------------------------------------------------

            parts = line.split(
                "\t",
                maxsplit=1
            )

            if len(parts) != 2:

                malformed.append({

                    "line_number":
                        line_number,

                    "reason":
                        "missing_tab_separator",

                    "content":
                        line,
                })

                continue

            image_name = parts[0].strip()
            label = parts[1].strip()

            # ------------------------------------------------
            # Empty filename
            # ------------------------------------------------

            if not image_name:

                malformed.append({

                    "line_number":
                        line_number,

                    "reason":
                        "empty_image_name",

                    "content":
                        line,
                })

                continue

            # ------------------------------------------------
            # Empty label
            # ------------------------------------------------

            if not label:

                malformed.append({

                    "line_number":
                        line_number,

                    "reason":
                        "empty_label",

                    "content":
                        line,
                })

                continue

            tokens = label.split()

            if len(tokens) == 0:

                malformed.append({

                    "line_number":
                        line_number,

                    "reason":
                        "zero_tokens",

                    "content":
                        line,
                })

                continue

            records.append({

                "image":
                    image_name,

                "label":
                    label,

                "line_number":
                    line_number,
            })

    return records, malformed


# ============================================================
# IMAGE HASH
# ============================================================

def image_hash(image_path):
    """
    Generate SHA-256 hash of decoded image content.

    This is used to detect exact image-content duplication.
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
        metadata
        +
        image.tobytes()
    ).hexdigest()


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(
    path,
    rows,
    fieldnames
):

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
# AUDIT ONE PARTITION
# ============================================================

def audit_partition(
    partition_name,
    image_dir,
    label_file
):

    print("\n" + "=" * 70)

    print(
        f"AUDITING CLEANED "
        f"{partition_name.upper()} DATA"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    records, malformed = read_label_file(
        label_file
    )

    # --------------------------------------------------------
    # Read image files
    # --------------------------------------------------------

    image_files = get_image_files(
        image_dir
    )

    # --------------------------------------------------------
    # Names from labels
    # --------------------------------------------------------

    label_names = {
        record["image"]
        for record in records
    }

    # --------------------------------------------------------
    # Names from images
    # --------------------------------------------------------

    image_names = set(
        image_files.keys()
    )

    # --------------------------------------------------------
    # Missing images
    # --------------------------------------------------------

    missing_images = sorted(
        label_names
        -
        image_names
    )

    # --------------------------------------------------------
    # Images without labels
    # --------------------------------------------------------

    images_without_labels = sorted(
        image_names
        -
        label_names
    )

    # --------------------------------------------------------
    # Duplicate filenames in labels
    # --------------------------------------------------------

    filename_records = defaultdict(list)

    for record in records:

        filename_records[
            record["image"]
        ].append(record)

    duplicate_label_names = {

        name: rows

        for name, rows
        in filename_records.items()

        if len(rows) > 1
    }

    # --------------------------------------------------------
    # Image hashes
    # --------------------------------------------------------

    hashes = {}

    corrupted_images = []

    dimensions = defaultdict(int)

    for filename, path in image_files.items():

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        # ----------------------------------------------------
        # Corrupted image
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
        # Hash
        # ----------------------------------------------------

        h = image_hash(path)

        if h is not None:

            hashes[filename] = h

    # --------------------------------------------------------
    # Hash groups
    # --------------------------------------------------------

    hash_groups = defaultdict(list)

    for filename, h in hashes.items():

        hash_groups[h].append(
            filename
        )

    # --------------------------------------------------------
    # Internal duplicate groups
    # --------------------------------------------------------

    duplicate_groups = {

        h: sorted(files)

        for h, files
        in hash_groups.items()

        if len(files) > 1
    }

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
    # Print
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
        f"{len(duplicate_label_names)}"
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

        "duplicate_label_names":
            duplicate_label_names,

        "corrupted_images":
            corrupted_images,

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
    }


# ============================================================
# TRAIN-TEST EXACT OVERLAP
# ============================================================

def find_train_test_overlap(
    train_audit,
    test_audit
):

    print("\n" + "=" * 70)

    print(
        "POST-CLEANING TRAIN-TEST "
        "EXACT IMAGE OVERLAP"
    )

    print("=" * 70)

    train_hash_groups = defaultdict(list)
    test_hash_groups = defaultdict(list)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for filename, h in (
        train_audit["hashes"].items()
    ):

        train_hash_groups[h].append(
            filename
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    for filename, h in (
        test_audit["hashes"].items()
    ):

        test_hash_groups[h].append(
            filename
        )

    # --------------------------------------------------------
    # Common hashes
    # --------------------------------------------------------

    common_hashes = sorted(
        set(train_hash_groups)
        &
        set(test_hash_groups)
    )

    overlap_groups = []

    for h in common_hashes:

        train_files = sorted(
            train_hash_groups[h]
        )

        test_files = sorted(
            test_hash_groups[h]
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
                (
                    len(train_files)
                    *
                    len(test_files)
                ),
        })

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

    total_pairings = sum(
        group["pair_count"]
        for group
        in overlap_groups
    )

    print(
        f"Remaining overlap groups : "
        f"{len(overlap_groups)}"
    )

    print(
        f"Train images involved     : "
        f"{len(unique_train_images)}"
    )

    print(
        f"Test images involved      : "
        f"{len(unique_test_images)}"
    )

    print(
        f"Total overlap pairings    : "
        f"{total_pairings}"
    )

    return overlap_groups


# ============================================================
# LABEL CONFLICT CHECK
# ============================================================

def find_label_conflicts(
    overlap_groups,
    train_records,
    test_records
):

    train_labels = defaultdict(list)
    test_labels = defaultdict(list)

    for record in train_records:

        train_labels[
            record["image"]
        ].append(
            record["label"]
        )

    for record in test_records:

        test_labels[
            record["image"]
        ].append(
            record["label"]
        )

    conflicts = []

    for group in overlap_groups:

        train_values = sorted({

            label

            for filename
            in group["train_images"]

            for label
            in train_labels.get(
                filename,
                []
            )
        })

        test_values = sorted({

            label

            for filename
            in group["test_images"]

            for label
            in test_labels.get(
                filename,
                []
            )
        })

        if (
            set(train_values)
            !=
            set(test_values)
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
                        train_values
                    ),

                "test_labels":
                    " || ".join(
                        test_values
                    ),
            })

    return conflicts


# ============================================================
# SAVE DUPLICATE REPORT
# ============================================================

def save_duplicate_report(
    train_audit,
    test_audit
):

    rows = []

    for partition, audit in [
        ("train", train_audit),
        ("test", test_audit)
    ]:

        for h, files in (
            audit["duplicate_groups"].items()
        ):

            files = sorted(files)

            for filename in files:

                rows.append({

                    "partition":
                        partition,

                    "hash":
                        h,

                    "group_size":
                        len(files),

                    "redundant_copies":
                        len(files) - 1,

                    "image":
                        filename,
                })

    write_csv(

        POST_DUPLICATES_CSV,

        rows,

        [
            "partition",
            "hash",
            "group_size",
            "redundant_copies",
            "image",
        ]
    )


# ============================================================
# SAVE OVERLAP REPORT
# ============================================================

def save_overlap_report(
    overlap_groups
):

    rows = []

    for group in overlap_groups:

        rows.append({

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

        POST_OVERLAP_CSV,

        rows,

        [
            "hash",
            "train_images",
            "test_images",
            "train_count",
            "test_count",
            "pair_count",
        ]
    )


# ============================================================
# SAVE CONFLICT REPORT
# ============================================================

def save_conflict_report(
    conflicts
):

    write_csv(

        POST_CONFLICTS_CSV,

        conflicts,

        [
            "hash",
            "train_images",
            "test_images",
            "train_labels",
            "test_labels",
        ]
    )


# ============================================================
# SAVE MISSING REPORT
# ============================================================

def save_missing_report(
    train_audit,
    test_audit
):

    rows = []

    for partition, audit in [
        ("train", train_audit),
        ("test", test_audit)
    ]:

        for filename in (
            audit["missing_images"]
        ):

            rows.append({

                "partition":
                    partition,

                "image":
                    filename,

                "reason":
                    "label_references_missing_image",
            })

        for filename in (
            audit["images_without_labels"]
        ):

            rows.append({

                "partition":
                    partition,

                "image":
                    filename,

                "reason":
                    "image_has_no_label",
            })

    write_csv(

        POST_MISSING_CSV,

        rows,

        [
            "partition",
            "image",
            "reason",
        ]
    )


# ============================================================
# SAVE MALFORMED REPORT
# ============================================================

def save_malformed_report(
    train_audit,
    test_audit
):

    rows = []

    for partition, audit in [
        ("train", train_audit),
        ("test", test_audit)
    ]:

        for item in audit["malformed"]:

            rows.append({

                "partition":
                    partition,

                "line_number":
                    item["line_number"],

                "reason":
                    item["reason"],

                "content":
                    item["content"],
            })

    write_csv(

        POST_MALFORMED_CSV,

        rows,

        [
            "partition",
            "line_number",
            "reason",
            "content",
        ]
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

def perform_final_checks(
    train_audit,
    test_audit,
    overlap_groups,
    conflicts
):

    checks = {}

    # --------------------------------------------------------
    # Train image-label count
    # --------------------------------------------------------

    checks[
        "train_image_label_count_match"
    ] = (
        len(train_audit["records"])
        ==
        len(train_audit["image_files"])
    )

    # --------------------------------------------------------
    # Test image-label count
    # --------------------------------------------------------

    checks[
        "test_image_label_count_match"
    ] = (
        len(test_audit["records"])
        ==
        len(test_audit["image_files"])
    )

    # --------------------------------------------------------
    # Missing train images
    # --------------------------------------------------------

    checks[
        "train_missing_images_zero"
    ] = (
        len(
            train_audit[
                "missing_images"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Missing test images
    # --------------------------------------------------------

    checks[
        "test_missing_images_zero"
    ] = (
        len(
            test_audit[
                "missing_images"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Train images without labels
    # --------------------------------------------------------

    checks[
        "train_images_without_labels_zero"
    ] = (
        len(
            train_audit[
                "images_without_labels"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Test images without labels
    # --------------------------------------------------------

    checks[
        "test_images_without_labels_zero"
    ] = (
        len(
            test_audit[
                "images_without_labels"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Corrupted train
    # --------------------------------------------------------

    checks[
        "train_corrupted_images_zero"
    ] = (
        len(
            train_audit[
                "corrupted_images"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Corrupted test
    # --------------------------------------------------------

    checks[
        "test_corrupted_images_zero"
    ] = (
        len(
            test_audit[
                "corrupted_images"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Duplicate train labels
    # --------------------------------------------------------

    checks[
        "train_duplicate_label_names_zero"
    ] = (
        len(
            train_audit[
                "duplicate_label_names"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Duplicate test labels
    # --------------------------------------------------------

    checks[
        "test_duplicate_label_names_zero"
    ] = (
        len(
            test_audit[
                "duplicate_label_names"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Internal train duplicates
    # --------------------------------------------------------

    checks[
        "train_internal_duplicate_groups_zero"
    ] = (
        len(
            train_audit[
                "duplicate_groups"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Internal test duplicates
    # --------------------------------------------------------

    checks[
        "test_internal_duplicate_groups_zero"
    ] = (
        len(
            test_audit[
                "duplicate_groups"
            ]
        )
        == 0
    )

    # --------------------------------------------------------
    # Most important check:
    # train-test exact image overlap
    # --------------------------------------------------------

    checks[
        "train_test_exact_overlap_zero"
    ] = (
        len(overlap_groups)
        == 0
    )

    # --------------------------------------------------------
    # Label conflicts
    #
    # If there is overlap, this should normally be zero as well.
    # --------------------------------------------------------

    checks[
        "remaining_label_conflicts_zero"
    ] = (
        len(conflicts)
        == 0
    )

    # --------------------------------------------------------
    # Expected train count
    # --------------------------------------------------------

    checks[
        "expected_train_count"
    ] = (
        len(train_audit["records"])
        ==
        EXPECTED_TRAIN_COUNT
    )

    # --------------------------------------------------------
    # Expected test count
    # --------------------------------------------------------

    checks[
        "expected_test_count"
    ] = (
        len(test_audit["records"])
        ==
        EXPECTED_TEST_COUNT
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    checks[
        "OVERALL_POST_CLEANING_AUDIT_PASS"
    ] = all(
        checks.values()
    )

    return checks


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(
    train_audit,
    test_audit,
    overlap_groups,
    conflicts,
    checks
):

    summary = {

        "dataset":
            "HME100K",

        "audit_type":
            "post_cleaning",

        "cleaned_dataset":
            str(
                CLEANED_DATASET_DIR
            ),

        "training": {

            "records":
                len(
                    train_audit["records"]
                ),

            "images":
                len(
                    train_audit["image_files"]
                ),

            "malformed":
                len(
                    train_audit["malformed"]
                ),

            "missing_images":
                len(
                    train_audit[
                        "missing_images"
                    ]
                ),

            "images_without_labels":
                len(
                    train_audit[
                        "images_without_labels"
                    ]
                ),

            "corrupted_images":
                len(
                    train_audit[
                        "corrupted_images"
                    ]
                ),

            "duplicate_label_names":
                len(
                    train_audit[
                        "duplicate_label_names"
                    ]
                ),

            "duplicate_groups":
                len(
                    train_audit[
                        "duplicate_groups"
                    ]
                ),

            "redundant_duplicate_images":
                train_audit[
                    "redundant_duplicate_images"
                ],
        },

        "testing": {

            "records":
                len(
                    test_audit["records"]
                ),

            "images":
                len(
                    test_audit["image_files"]
                ),

            "malformed":
                len(
                    test_audit["malformed"]
                ),

            "missing_images":
                len(
                    test_audit[
                        "missing_images"
                    ]
                ),

            "images_without_labels":
                len(
                    test_audit[
                        "images_without_labels"
                    ]
                ),

            "corrupted_images":
                len(
                    test_audit[
                        "corrupted_images"
                    ]
                ),

            "duplicate_label_names":
                len(
                    test_audit[
                        "duplicate_label_names"
                    ]
                ),

            "duplicate_groups":
                len(
                    test_audit[
                        "duplicate_groups"
                    ]
                ),

            "redundant_duplicate_images":
                test_audit[
                    "redundant_duplicate_images"
                ],
        },

        "train_test_overlap": {

            "remaining_unique_overlap_groups":
                len(overlap_groups),

            "remaining_overlap_pairings":
                sum(
                    group["pair_count"]
                    for group in overlap_groups
                ),

            "remaining_label_conflicts":
                len(conflicts),
        },

        "expected_counts": {

            "original_train":
                ORIGINAL_TRAIN_COUNT,

            "original_test":
                ORIGINAL_TEST_COUNT,

            "expected_train_removed":
                EXPECTED_TRAIN_REMOVED,

            "expected_test_removed":
                EXPECTED_TEST_REMOVED,

            "expected_clean_train":
                EXPECTED_TRAIN_COUNT,

            "expected_clean_test":
                EXPECTED_TEST_COUNT,
        },

        "verification":
            checks,
    }

    # ========================================================
    # JSON
    # ========================================================

    with open(
        POST_AUDIT_SUMMARY_JSON,
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
    # TXT
    # ========================================================

    with open(
        POST_AUDIT_SUMMARY_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 75 + "\n"
        )

        f.write(
            "HME100K POST-CLEANING DATASET AUDIT\n"
        )

        f.write(
            "=" * 75 + "\n\n"
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        f.write(
            "CLEANED DATASET\n"
        )

        f.write(
            f"{CLEANED_DATASET_DIR}\n\n"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        f.write(
            "TRAINING DATA\n"
        )

        f.write(
            f"Records              : "
            f"{len(train_audit['records'])}\n"
        )

        f.write(
            f"Images               : "
            f"{len(train_audit['image_files'])}\n"
        )

        f.write(
            f"Malformed records    : "
            f"{len(train_audit['malformed'])}\n"
        )

        f.write(
            f"Missing images       : "
            f"{len(train_audit['missing_images'])}\n"
        )

        f.write(
            f"Images without labels: "
            f"{len(train_audit['images_without_labels'])}\n"
        )

        f.write(
            f"Corrupted images     : "
            f"{len(train_audit['corrupted_images'])}\n"
        )

        f.write(
            f"Duplicate label names: "
            f"{len(train_audit['duplicate_label_names'])}\n"
        )

        f.write(
            f"Duplicate image groups: "
            f"{len(train_audit['duplicate_groups'])}\n"
        )

        f.write(
            f"Redundant duplicates : "
            f"{train_audit['redundant_duplicate_images']}\n\n"
        )

        # ----------------------------------------------------
        # Testing
        # ----------------------------------------------------

        f.write(
            "TESTING DATA\n"
        )

        f.write(
            f"Records              : "
            f"{len(test_audit['records'])}\n"
        )

        f.write(
            f"Images               : "
            f"{len(test_audit['image_files'])}\n"
        )

        f.write(
            f"Malformed records    : "
            f"{len(test_audit['malformed'])}\n"
        )

        f.write(
            f"Missing images       : "
            f"{len(test_audit['missing_images'])}\n"
        )

        f.write(
            f"Images without labels: "
            f"{len(test_audit['images_without_labels'])}\n"
        )

        f.write(
            f"Corrupted images     : "
            f"{len(test_audit['corrupted_images'])}\n"
        )

        f.write(
            f"Duplicate label names: "
            f"{len(test_audit['duplicate_label_names'])}\n"
        )

        f.write(
            f"Duplicate image groups: "
            f"{len(test_audit['duplicate_groups'])}\n"
        )

        f.write(
            f"Redundant duplicates : "
            f"{test_audit['redundant_duplicate_images']}\n\n"
        )

        # ----------------------------------------------------
        # Leakage
        # ----------------------------------------------------

        f.write(
            "TRAIN-TEST EXACT IMAGE OVERLAP\n"
        )

        f.write(
            f"Remaining overlap groups : "
            f"{len(overlap_groups)}\n"
        )

        f.write(
            f"Remaining overlap pairs  : "
            f"{sum(
                group['pair_count']
                for group in overlap_groups
            )}\n"
        )

        f.write(
            f"Remaining label conflicts: "
            f"{len(conflicts)}\n\n"
        )

        # ----------------------------------------------------
        # Expected counts
        # ----------------------------------------------------

        f.write(
            "EXPECTED COUNTS\n"
        )

        f.write(
            f"Expected clean train : "
            f"{EXPECTED_TRAIN_COUNT}\n"
        )

        f.write(
            f"Actual clean train   : "
            f"{len(train_audit['records'])}\n"
        )

        f.write(
            f"Expected clean test  : "
            f"{EXPECTED_TEST_COUNT}\n"
        )

        f.write(
            f"Actual clean test    : "
            f"{len(test_audit['records'])}\n\n"
        )

        # ----------------------------------------------------
        # Verification
        # ----------------------------------------------------

        f.write(
            "VERIFICATION CHECKS\n"
        )

        for name, value in checks.items():

            status = (
                "PASS"
                if value
                else "FAIL"
            )

            f.write(
                f"{name:<50} : "
                f"{status}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Final status
        # ----------------------------------------------------

        if checks[
            "OVERALL_POST_CLEANING_AUDIT_PASS"
        ]:

            f.write(
                "FINAL STATUS: PASS\n"
            )

            f.write(
                "The cleaned dataset passed all "
                "post-cleaning integrity and "
                "train-test separation checks.\n"
            )

        else:

            f.write(
                "FINAL STATUS: FAIL\n"
            )

            f.write(
                "One or more post-cleaning checks "
                "require investigation before "
                "model training.\n"
            )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 75)

    print(
        "HME100K POST-CLEANING DATASET AUDIT"
    )

    print("=" * 75)

    print(
        f"\nCleaned dataset:"
        f"\n{CLEANED_DATASET_DIR}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script ONLY audits the dataset."
    )

    print(
        "It will NOT modify or delete any files."
    )

    # ========================================================
    # Verify directories
    # ========================================================

    if not CLEANED_DATASET_DIR.exists():

        raise FileNotFoundError(
            f"\nCleaned dataset not found:\n"
            f"{CLEANED_DATASET_DIR}\n\n"
            f"Run clean_dataset.py first."
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
    # Exact train-test overlap
    # ========================================================

    overlap_groups = find_train_test_overlap(
        train_audit,
        test_audit
    )

    # ========================================================
    # Label conflicts
    # ========================================================

    conflicts = find_label_conflicts(
        overlap_groups,
        train_audit["records"],
        test_audit["records"]
    )

    print(
        f"Remaining label conflicts: "
        f"{len(conflicts)}"
    )

    # ========================================================
    # Final checks
    # ========================================================

    checks = perform_final_checks(
        train_audit,
        test_audit,
        overlap_groups,
        conflicts
    )

    # ========================================================
    # Save reports
    # ========================================================

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_duplicate_report(
        train_audit,
        test_audit
    )

    save_overlap_report(
        overlap_groups
    )

    save_conflict_report(
        conflicts
    )

    save_missing_report(
        train_audit,
        test_audit
    )

    save_malformed_report(
        train_audit,
        test_audit
    )

    summary = create_summary(
        train_audit,
        test_audit,
        overlap_groups,
        conflicts,
        checks
    )

    # ========================================================
    # Print verification
    # ========================================================

    print("\n" + "=" * 75)

    print(
        "POST-CLEANING VERIFICATION"
    )

    print("=" * 75)

    for name, value in checks.items():

        status = (
            "PASS"
            if value
            else "FAIL"
        )

        print(
            f"{name:<55} : {status}"
        )

    # ========================================================
    # Final result
    # ========================================================

    print("\n" + "=" * 75)

    if checks[
        "OVERALL_POST_CLEANING_AUDIT_PASS"
    ]:

        print(
            "POST-CLEANING AUDIT: PASS"
        )

        print("=" * 75)

        print(
            "\nThe cleaned dataset passed all "
            "verification checks."
        )

        print(
            "\nMost important result:"
        )

        print(
            "Remaining exact train-test image "
            "overlap = 0"
        )

        print(
            "\nThe dataset is ready for the "
            "next vocabulary-preparation step."
        )

    else:

        print(
            "POST-CLEANING AUDIT: FAIL"
        )

        print("=" * 75)

        print(
            "\nDo NOT start model training."
        )

        print(
            "Review the failed checks first."
        )

    # ========================================================
    # Report locations
    # ========================================================

    print(
        f"\nDetailed reports saved in:"
        f"\n{AUDIT_DIR}"
    )

    print(
        f"\nJSON:"
        f"\n{POST_AUDIT_SUMMARY_JSON}"
    )

    print(
        f"\nTXT:"
        f"\n{POST_AUDIT_SUMMARY_TXT}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()