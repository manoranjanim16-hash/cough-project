from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CHECKPOINT, LABELS, SAMPLE_RATE
from .cough_detector import CoughSegment, detect_cough_segments
from .model import CoughCNN
from .preprocess import audio_to_mel, load_audio


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: np.ndarray
    segments: list[CoughSegment]


def load_checkpoint(path: str | Path | None = None) -> CoughCNN:
    model = CoughCNN(num_classes=len(LABELS))
    checkpoint = Path(path) if path else DEFAULT_CHECKPOINT
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
    model.eval()
    return model


def predict_audio(audio: np.ndarray, checkpoint: str | Path | None = None) -> Prediction:
    segments = detect_cough_segments(audio, SAMPLE_RATE)
    model = load_checkpoint(checkpoint)
    mel = torch.from_numpy(audio_to_mel(audio)).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(mel), dim=1)[0].numpy()
    index = int(probabilities.argmax())
    return Prediction(LABELS[index], probabilities, segments)


def predict_file(path: str | Path, checkpoint: str | Path | None = None) -> Prediction:
    return predict_audio(load_audio(str(path)), checkpoint)


def predict_bytes(data: bytes, checkpoint: str | Path | None = None) -> Prediction:
    return predict_audio(load_audio(data), checkpoint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict a cough recording")
    parser.add_argument("audio")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    result = predict_file(args.audio, args.checkpoint)
    print(f"label={result.label}")
    print(" ".join(f"{label}={value:.4f}" for label, value in zip(LABELS, result.probabilities)))
