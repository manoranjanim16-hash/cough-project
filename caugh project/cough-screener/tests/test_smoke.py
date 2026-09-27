import numpy as np
import torch

from src.cough_detector import detect_cough_segments
from src.model import CoughCNN, parameter_count
from src.preprocess import audio_to_mel


def test_model_and_preprocess_shapes() -> None:
    audio = np.zeros(32_000, dtype=np.float32)
    audio[8_000:12_000] = 0.8 * np.sin(np.linspace(0, 400 * np.pi, 4_000))
    mel = audio_to_mel(audio)
    model = CoughCNN()
    output = model(torch.from_numpy(mel).unsqueeze(0).unsqueeze(0))
    assert mel.shape[0] == 96
    assert output.shape == (1, 4)
    assert 100_000 <= parameter_count(model) <= 110_000
    assert detect_cough_segments(audio)
