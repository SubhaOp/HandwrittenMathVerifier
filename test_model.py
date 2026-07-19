import torch

from src.model import Encoder

model = Encoder()

x = torch.randn(2, 3, 128, 512)

y = model(x)

print(y.shape)