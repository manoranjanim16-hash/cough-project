from __future__ import annotations

from io import BytesIO

import librosa
import numpy as np
import soundfile as sf

from .config import CLIP_SAMPLES, FMAX, FMIN, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


def load_audio(source: str | bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if isinstance(source, bytes):
        audio, rate = sf.read(BytesIO(source), dtype="float32", always_2d=False)
    else:
        audio, rate = sf.read(source, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sample_rate)
    peak = np.max(np.abs(audio), initial=0.0)
    return (audio / peak).astype(np.float32) if peak > 1e-8 else audio


def pad_or_crop(audio: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    if audio.size >= length:
        start = (audio.size - length) // 2
        return audio[start : start + length]
    return np.pad(audio, (0, length - audio.size))


def audio_to_mel(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    clip = pad_or_crop(np.asarray(audio, dtype=np.float32))
    mel = librosa.feature.melspectrogram(
        y=clip, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=min(FMAX, sample_rate // 2), power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return ((mel_db + 80.0) / 80.0).astype(np.float32)
