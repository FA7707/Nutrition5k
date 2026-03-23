"""Multi-task food recognition model.

Architecture:
    ResNet-50 backbone (ImageNet pretrained) -> shared features (2048-d)
        -> Classification head: ingredient multi-label prediction
        -> Regression head: calories, fat, carbs, protein estimation
"""

import torch
import torch.nn as nn
from torchvision import models


class NutritionModel(nn.Module):
    """Multi-task model for food image classification and nutrition estimation.

    Args:
        num_classes: Number of ingredient classes for multi-label classification.
        pretrained: Whether to use ImageNet pretrained backbone.
        dropout: Dropout rate for head layers.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: ResNet-50 without the final FC layer
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # -> [B, 2048, 1, 1]
        self.feature_dim = 2048

        # Classification head — multi-label ingredient prediction
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        # Regression head — predict [calories, fat, carbs, protein]
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images [B, 3, 224, 224].

        Returns:
            cls_logits: [B, num_classes] — raw logits for multi-label classification.
            reg_output: [B, 4] — predicted (calories, fat, carbs, protein).
        """
        features = self.backbone(x)  # [B, 2048, 1, 1]
        cls_logits = self.cls_head(features)
        reg_output = self.reg_head(features)
        return cls_logits, reg_output

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning warmup."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
