# VoiceFilter Cloud Server

Lightweight Flask API for speaker voice isolation using HuggingFace Target Speaker Extraction.

## Features

- Extracts target speaker's voice from noisy/mixed audio
- Uses reference audio to identify the target speaker
- Powered by HuggingFace API (no local GPU needed)

## API Endpoints

- `GET /` - Server info
- `GET /health` - Health check
- `POST /filter-with-reference` - Extract target speaker
  - `noisy`: WAV file with mixed audio
  - `reference`: WAV file of target speaker's voice
  - Returns: WAV file with extracted voice

## Deployment

### Railway (Recommended)

1. Connect this repo to Railway
2. Railway auto-detects Python and deploys
3. Use the provided URL in your app

### Local Development

```bash
pip install -r requirements.txt
python server_cloud.py
```

Server runs on `http://localhost:5001`

## Environment Variables

- `PORT` - Server port (default: 5001)
