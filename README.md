---
title: VoiceFilter API
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# VoiceFilter API - Speaker Voice Isolation

REST API for extracting a target speaker's voice from noisy/mixed audio.

## API Endpoints

- `GET /health` - Health check
- `POST /filter-with-reference` - Extract voice (multipart: `noisy` + `reference` WAV files)

## Usage

```bash
curl -X POST https://YOUR-SPACE.hf.space/filter-with-reference \
  -F "noisy=@noisy_audio.wav" \
  -F "reference=@reference.wav" \
  --output enhanced.wav
```
