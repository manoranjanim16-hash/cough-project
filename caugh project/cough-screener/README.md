# Cough Screener

A research/demo Streamlit app for exploring cough audio classification. Create a lightweight patient or study profile, upload a WAV/MP3/M4A recording or use the browser microphone, then review detected cough segments and model estimates in a session history.

> **Medical disclaimer:** This is not a medical device and must not be used to diagnose, treat, or rule out any condition. Results are experimental, may be wrong, and are especially unreliable outside the data distribution used for training. Seek advice from a qualified clinician for health concerns. Do not upload personally identifiable or sensitive recordings to an untrusted deployment.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The app runs without a trained checkpoint by using a clearly marked randomly initialized model. To use a real model, place a checkpoint at `artifacts/cough_cnn.pt` or pass `--checkpoint` to the inference CLI.

## App features

- Dashboard with assessment and model status summaries.
- Patient/study profile fields for ID, alias, age, gender, weight, height, smoking status, symptoms, and research notes.
- Upload or browser microphone recording workflow.
- Detected cough segment timing and model probability chart.
- Session-only patient history with filtering and CSV export.
- Clear demo-mode and medical-disclaimer framing.

Patient history is held in Streamlit session memory and is cleared when the app restarts. No database, authentication, encryption, or cloud storage is included. Use study IDs and avoid identifiable health information during experimentation.

## Training

Arrange labeled WAV files as follows:

```text
data/audio/
  healthy/*.wav
  symptomatic/*.wav
  asthma/*.wav
  copd/*.wav
```

Then run:

```bash
python -m src.train --data-dir data/audio --output artifacts/cough_cnn.pt --epochs 20
```

The training script uses class-weighted cross entropy, deterministic validation splitting, SpecAugment, and saves the best validation checkpoint.

## CLI inference

```bash
python -m src.infer recording.wav --checkpoint artifacts/cough_cnn.pt
```

## Scope and limitations

This repository provides an educational baseline, not a validated clinical system. Dataset licenses, consent, demographic coverage, recording conditions, and label quality must be reviewed before any research or deployment use. See `data/README.md` for source dataset links and the expected folder layout.
