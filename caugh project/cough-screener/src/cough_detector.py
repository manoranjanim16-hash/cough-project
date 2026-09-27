from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label

from .config import (
    SAMPLE_RATE,
    SEGMENT_MAX_SECONDS,
    SEGMENT_MIN_SECONDS,
    SEGMENT_PAD_SECONDS,
)


@dataclass(frozen=True)
class CoughSegment:
    start: float
    end: float


def detect_cough_segments(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[CoughSegment]:
    """Find energetic, compact regions that are plausible cough candidates."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return []
    frame_length = max(1, int(0.025 * sample_rate))
    hop = max(1, int(0.010 * sample_rate))
    padded = np.pad(samples, (0, max(0, frame_length - samples.size)))
    frame_count = 1 + max(0, (padded.size - frame_length) // hop)
    frames = np.stack([padded[index * hop : index * hop + frame_length] for index in range(frame_count)])
    energy = np.sqrt(np.mean(frames * frames, axis=1) + 1e-8)
    threshold = max(float(np.percentile(energy, 70)) * 1.35, float(energy.mean() + energy.std()))
    active = energy > threshold
    active = binary_closing(binary_opening(active, structure=np.ones(3)), structure=np.ones(5))
    groups, count = label(active)
    segments: list[CoughSegment] = []
    for group in range(1, count + 1):
        indices = np.flatnonzero(groups == group)
        if indices.size == 0:
            continue
        start = max(0.0, float(indices[0] * hop / sample_rate - SEGMENT_PAD_SECONDS))
        end = min(float(samples.size / sample_rate), float((indices[-1] * hop + frame_length) / sample_rate + SEGMENT_PAD_SECONDS))
        duration = end - start
        if SEGMENT_MIN_SECONDS <= duration <= SEGMENT_MAX_SECONDS:
            segments.append(CoughSegment(start, end))
    return segments
