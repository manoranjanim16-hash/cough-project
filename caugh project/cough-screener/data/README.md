# Data layout

This project does not bundle audio data. Review each source's terms, consent requirements, and label definitions before use.

Potential research sources:

- [Coswara](https://github.com/iiscleap/Coswara-Data): respiratory sound and symptom metadata.
- [COUGHVID](https://zenodo.org/records/7024894): crowdsourced cough recordings and annotations.
- [Virufy](https://github.com/virufy/virufy-data): respiratory audio research data.

For the included training loader, convert or copy WAV files into:

```text
data/audio/
  healthy/*.wav
  symptomatic/*.wav
  asthma/*.wav
  copd/*.wav
```

The folder names must exactly match the labels in `src/config.py`. Do not commit private or identifiable recordings.
