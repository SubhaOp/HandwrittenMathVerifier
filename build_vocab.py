import json
from pathlib import Path

# =====================================================
# Read all characters from training labels
# =====================================================

chars = set()

with open(
    "dataset/train/train_labels.txt",
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

        chars.update(label)

# Sort characters
chars = sorted(list(chars))

# =====================================================
# Reserve index 0 for CTC Blank
# =====================================================

char2idx = {}

idx2char = {}

# Blank token
idx2char[0] = "<BLANK>"

# Start characters from index 1
for i, c in enumerate(chars, start=1):
    char2idx[c] = i
    idx2char[i] = c

# =====================================================
# Save vocabulary
# =====================================================

Path("configs").mkdir(exist_ok=True)

with open(
    "configs/char2idx.json",
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
    "configs/idx2char.json",
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
print("Vocabulary Built Successfully")
print("=" * 60)

print("Total Characters :", len(chars))
print("Total Classes    :", len(chars) + 1)
print("Blank Index      : 0")

print("\nFirst 10 mappings:\n")

for k, v in list(char2idx.items())[:10]:
    print(f"{repr(k):>8} -> {v}")

print("\nFiles Saved:")
print("configs/char2idx.json")
print("configs/idx2char.json")