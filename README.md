---
title: VoiceFilter API
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# VoiceFilter - Speaker Voice Isolation

Extract a target speaker's voice from noisy/mixed audio using a reference sample.

## How it works

1. Upload noisy audio (mixed voices or background noise)
2. Upload reference audio (clean sample of target speaker)
3. Get extracted voice of just the target speaker

## API Usage

This Space provides an API endpoint that can be called programmatically:

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/voicefilter-api")
result = client.predict(
    "noisy_audio.wav",
    "reference_audio.wav",
    api_name="/predict"
)
print(result)  # Path to extracted audio
```

Powered by [Target Speaker Extraction](https://huggingface.co/spaces/swc2/Target-speaker-extraction)
