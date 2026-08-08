import torch
import torch.nn as nn
import torchvision.models as models


class MathRecognizer(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # ==================================================
        # ResNet18 Backbone
        # ==================================================

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # ==================================================
        # IMPORTANT V3 CHANGE
        #
        # Preserve more horizontal resolution.
        #
        # Input:
        #     (B, 3, 128, 1536)
        #
        # V2:
        #     approximately 192 horizontal steps
        #
        # V3:
        #     approximately 384 horizontal steps
        #
        # This gives CTC much more sequence capacity for
        # long mathematical expressions.
        # ==================================================

        # --------------------------------------------------
        # Layer 2
        # --------------------------------------------------

        backbone.layer2[0].conv1.stride = (2, 1)

        backbone.layer2[0].downsample[0].stride = (2, 1)

        # --------------------------------------------------
        # Layer 3
        # --------------------------------------------------

        backbone.layer3[0].conv1.stride = (2, 1)

        backbone.layer3[0].downsample[0].stride = (2, 1)

        # --------------------------------------------------
        # Layer 4
        # --------------------------------------------------

        backbone.layer4[0].conv1.stride = (2, 1)

        backbone.layer4[0].downsample[0].stride = (2, 1)

        # ==================================================
        # Remove average pooling and classification layer
        # ==================================================

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        # ==================================================
        # Height Pooling
        # ==================================================

        # Collapse only the height dimension.
        #
        # Width remains the CTC sequence dimension.

        self.height_pool = nn.AdaptiveAvgPool2d(
            (1, None)
        )

        # ==================================================
        # BiLSTM
        # ==================================================

        self.lstm = nn.LSTM(

            input_size=512,

            hidden_size=256,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.3
        )

        # ==================================================
        # Classifier
        # ==================================================

        self.dropout = nn.Dropout(
            0.3
        )

        # 256 forward + 256 backward = 512

        self.fc = nn.Linear(
            512,
            num_classes
        )

    # ==================================================
    # Forward
    # ==================================================

    def forward(self, x):

        # --------------------------------------------------
        # CNN feature extraction
        # --------------------------------------------------

        x = self.cnn(x)

        # Expected V3:
        #
        # Input:
        # (B, 3, 128, 1536)
        #
        # Output approximately:
        # (B, 512, 4, 384)

        # --------------------------------------------------
        # Collapse height
        # --------------------------------------------------

        x = self.height_pool(x)

        # (B, 512, 1, 384)

        x = x.squeeze(2)

        # (B, 512, 384)

        # --------------------------------------------------
        # Width -> sequence
        # --------------------------------------------------

        x = x.permute(
            0,
            2,
            1
        )

        # (B, 384, 512)

        # --------------------------------------------------
        # BiLSTM
        # --------------------------------------------------

        self.lstm.flatten_parameters()

        x, _ = self.lstm(x)

        # (B, 384, 512)

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        x = self.dropout(x)

        x = self.fc(x)

        # (B, 384, num_classes)

        return x