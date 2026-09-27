from __future__ import annotations

from pathlib import Path

LABELS = ("healthy", "symptomatic", "asthma", "copd")
SAMPLE_RATE = 16_000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_MELS = 96
N_FFT = 512
HOP_LENGTH = 160
FMIN = 50
FMAX = 8_000
SEGMENT_MIN_SECONDS = 0.08
SEGMENT_MAX_SECONDS = 2.0
SEGMENT_PAD_SECONDS = 0.08
MODEL_INPUT_SHAPE = (N_MELS, 1 + CLIP_SAMPLES // HOP_LENGTH)
DEFAULT_CHECKPOINT = Path("artifacts/cough_cnn.pt")
