import json
from collections import Counter

from src.config import (
    TRAIN_LABEL_FILE,
    CHAR2IDX_FILE,
    IDX2CHAR_FILE,
    CONFIG_DIR,
)

# =====================================================
# Read all SYMBOLS (whitespace-separated tokens) from
# training labels
# =====================================================
#
# CHANGED: token-level instead of character-level.
#
# train_labels.txt stores labels as space-separated
# symbols, e.g.:
#     1 0 \times ( x + 5 ) = 1 3 \times ( \frac { 5 } { 7 } x + 5 )
#
# The old chars.update(label) split every multi-character
# command (\frac, \times, \angle, \sqrt, ...) into individual
# letters with no visual grounding in the image -- \frac alone
# appears in 64% of train_labels.txt. tokens.update(label.split())
# keeps each symbol atomic instead.
#
# Also switched from a hardcoded relative path to src.config's
# TRAIN_LABEL_FILE, which was silently pointing at the wrong
# location on Colab (config.py resolves dataset paths per-
# environment; this file didn't use it).

tokens = Counter()

with open(
    TRAIN_LABEL_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) < 2:
            continue

        label = parts[1]

        tokens.update(label.split())

# Sort symbols
vocab = sorted(tokens.keys())

# =====================================================
# Reserve index 0 for CTC Blank
# =====================================================

char2idx = {}

idx2char = {}

# Blank token
idx2char[0] = "<BLANK>"

# Start symbols from index 1
for i, sym in enumerate(vocab, start=1):
    char2idx[sym] = i
    idx2char[i] = sym

# =====================================================
# Save vocabulary
# =====================================================

CONFIG_DIR.mkdir(parents=True, exist_ok=True)

with open(
    CHAR2IDX_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        char2idx,
        f,
        indent=4,
        ensure_ascii=False
    )

with open(
    IDX2CHAR_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        idx2char,
        f,
        indent=4,
        ensure_ascii=False
    )

# =====================================================
# Print information
# =====================================================

print("=" * 60)
print("Vocabulary Built Successfully (token-level)")
print("=" * 60)

print("Total Symbols :", len(vocab))
print("Total Classes :", len(vocab) + 1)
print("Blank Index   : 0")

print("\nFirst 10 mappings:\n")

for k, v in list(char2idx.items())[:10]:
    print(f"{repr(k):>15} -> {v}")

print("\n20 most common symbols:\n")

for sym, count in tokens.most_common(20):
    print(f"{repr(sym):>15} : {count}")

print("\nFiles Saved:")
print(CHAR2IDX_FILE)
print(IDX2CHAR_FILE)