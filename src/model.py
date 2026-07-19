import torch
import torch.nn as nn
import torchvision.models as models


class MathRecognizer(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.lstm = nn.LSTM(
            input_size=512 * 4,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        x = self.cnn(x)

        b, c, h, w = x.shape

        x = x.permute(0, 3, 1, 2)

        x = x.reshape(b, w, c * h)

        x, _ = self.lstm(x)

        x = self.fc(x)

        return x