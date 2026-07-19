import json

chars = set()

with open("dataset/train/train_labels.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) < 2:
            continue

        label = parts[1]

        chars.update(label)

chars = sorted(list(chars))

char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

with open("configs/char2idx.json", "w") as f:
    json.dump(char2idx, f, indent=4)

with open("configs/idx2char.json", "w") as f:
    json.dump(idx2char, f, indent=4)

print("Vocabulary size:", len(char2idx))
print(chars)