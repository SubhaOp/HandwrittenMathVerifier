import torch
import torch.nn as nn
import torchvision.models as models


class MathRecognizer(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # ==========================================
        # ResNet18 Backbone
        # ==========================================

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # --------------------------------------------------
        # IMPORTANT:
        # Standard ResNet18 reduces both H and W by 32x.
        #
        # Input:
        # (B, 3, 128, 1536)
        #
        # Standard output:
        # (B, 512, 4, 48)
        #
        # For CTC, 48 time steps are too few.
        #
        # We therefore prevent horizontal downsampling
        # in layer3 and layer4 while still reducing height.
        # --------------------------------------------------

        # layer3 first block
        backbone.layer3[0].conv1.stride = (2, 1)
        backbone.layer3[0].downsample[0].stride = (2, 1)

        # layer4 first block
        backbone.layer4[0].conv1.stride = (2, 1)
        backbone.layer4[0].downsample[0].stride = (2, 1)

        # Remove global average pooling and FC
        self.cnn = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        # ==========================================
        # Height Pooling
        # ==========================================

        # Collapse only the height dimension.
        # Width is preserved as the CTC sequence.
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        # ==========================================
        # BiLSTM
        # ==========================================

        self.lstm = nn.LSTM(
            input_size=512,
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

        # Bidirectional:
        # 256 forward + 256 backward = 512
        self.fc = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        # ==========================================
        # CNN Feature Extraction
        # ==========================================

        x = self.cnn(x)

        # Expected:
        # (B, 512, 4, 192)
        # for input (B, 3, 128, 1536)

        # ==========================================
        # Collapse Height
        # ==========================================

        x = self.height_pool(x)

        # (B, 512, 1, 192)

        x = x.squeeze(2)

        # (B, 512, 192)

        # ==========================================
        # Convert width to sequence
        # ==========================================

        x = x.permute(0, 2, 1)

        # (B, 192, 512)

        # ==========================================
        # BiLSTM
        # ==========================================

        self.lstm.flatten_parameters()

        x, _ = self.lstm(x)

        # (B, 192, 512)

        # ==========================================
        # Classification
        # ==========================================

        x = self.dropout(x)

        x = self.fc(x)

        # (B, 192, num_classes)

        return x