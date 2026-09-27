from __future__ import annotations

import torch
from torch import nn


class CoughCNN(nn.Module):
    """Compact mel-spectrogram CNN with approximately 106K parameters."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 113, 3, padding=1), nn.BatchNorm2d(113), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(113, 144), nn.ReLU(), nn.Dropout(0.2), nn.Linear(144, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parameter_count(model: nn.Module | None = None) -> int:
    return sum(parameter.numel() for parameter in (model or CoughCNN()).parameters())
