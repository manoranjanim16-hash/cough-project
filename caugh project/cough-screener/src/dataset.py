from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import LABELS
from .preprocess import audio_to_mel, load_audio


class CoughDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, files: list[Path], augment: bool = False) -> None:
        self.files = files
        self.augment = augment
        self.label_ids = {label: index for index, label in enumerate(LABELS)}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path = self.files[index]
        audio = load_audio(str(path))
        mel = audio_to_mel(audio)
        if self.augment:
            mel = spec_augment(mel)
        label = self.label_ids[path.parent.name.lower()]
        return torch.from_numpy(mel).unsqueeze(0), torch.tensor(label, dtype=torch.long)


def spec_augment(mel: np.ndarray, max_time_mask: int = 12, max_freq_mask: int = 10) -> np.ndarray:
    result = mel.copy()
    time_width = np.random.randint(0, min(max_time_mask, result.shape[1]) + 1)
    freq_width = np.random.randint(0, min(max_freq_mask, result.shape[0]) + 1)
    if time_width:
        start = np.random.randint(0, result.shape[1] - time_width + 1)
        result[:, start : start + time_width] = 0.0
    if freq_width:
        start = np.random.randint(0, result.shape[0] - freq_width + 1)
        result[start : start + freq_width, :] = 0.0
    return result


def discover_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    return sorted(path for label in LABELS for path in (root / label).glob("*.wav"))
