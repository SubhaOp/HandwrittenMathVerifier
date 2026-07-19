import json
import torch

from src.model import MathRecognizer

with open("configs/char2idx.json") as f:
    vocab = json.load(f)

num_classes = len(vocab) + 1

model = MathRecognizer(num_classes)

x = torch.randn(2, 3, 128, 512)

y = model(x)

print(y.shape)