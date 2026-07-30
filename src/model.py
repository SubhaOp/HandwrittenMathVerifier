import torch
import torch.nn as nn
import torchvision.models as models


class MathRecognizer(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # ==========================================
        # CNN Backbone
        # ==========================================

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Remove Average Pool and FC layers
        self.cnn = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        # ==========================================
        # BiLSTM
        # ==========================================

        self.lstm = nn.LSTM(
            input_size=512 * 4,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # ==========================================
        # Classifier
        # ==========================================

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        # CNN Feature Extraction
        x = self.cnn(x)

        # Shape:
        # (B, 512, 4, W)

        b, c, h, w = x.size()

        # Convert width dimension into sequence
        x = x.permute(0, 3, 1, 2)

        # (B, W, 512*4)
        x = x.reshape(b, w, c * h)

        # Better CUDA performance
        self.lstm.flatten_parameters()

        # BiLSTM
        x, _ = self.lstm(x)

        # Dropout
        x = self.dropout(x)

        # Character classification
        x = self.fc(x)

        return x