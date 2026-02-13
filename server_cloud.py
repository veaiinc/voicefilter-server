"""
VoiceFilter Cloud Server - Lightweight Flask API for speaker voice isolation
Uses HuggingFace API only (no local models needed).
"""

import os
import tempfile
import shutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io

# Import Gradio client for HuggingFace API
try:
    from gradio_client import Client, handle_file
    HF_CLIENT_AVAILABLE = True
except ImportError:
    HF_CLIENT_AVAILABLE = False
    print("ERROR: gradio_client not installed!")

app = Flask(__name__)
CORS(app)

# HuggingFace client
hf_tse_client = None

def init_client():
    """Initialize HuggingFace API client"""
    global hf_tse_client

    if not HF_CLIENT_AVAILABLE:
        print("gradio_client not available")
        return

    try:
        hf_tse_client = Client("swc2/Target-speaker-extraction")
        print("HuggingFace Target Speaker Extraction API connected")
    except Exception as e:
        print(f"Failed to connect to HuggingFace API: {e}")
        hf_tse_client = None

@app.route('/', methods=['GET'])
def index():
    """Homepage"""
    return jsonify({
        'service': 'VoiceFilter Cloud API',
        'status': 'running',
        'hf_connected': hf_tse_client is not None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'hf_tse_available': hf_tse_client is not None,
        'embedder_loaded': hf_tse_client is not None,
        'voicefilter_loaded': hf_tse_client is not None
    })

@app.route('/filter-with-reference', methods=['POST'])
def filter_with_reference():
    """
    Extract target speaker using reference audio via HuggingFace API
    """
    if hf_tse_client is None:
        return jsonify({'error': 'HuggingFace API not available'}), 500

    if 'noisy' not in request.files or 'reference' not in request.files:
        return jsonify({'error': 'Both noisy and reference audio files required'}), 400

    try:
        # Save uploaded files to temp
        noisy_file = request.files['noisy']
        ref_file = request.files['reference']

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            noisy_file.save(tmp.name)
            noisy_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            ref_file.save(tmp.name)
            ref_path = tmp.name

        print(f"Processing: noisy={os.path.getsize(noisy_path)} bytes, ref={os.path.getsize(ref_path)} bytes")

        # Call HuggingFace API
        result = hf_tse_client.predict(
            handle_file(noisy_path),
            handle_file(ref_path),
            "mix",
            "iter_model",
            api_name="/gradio_TSE"
        )

        # Result[1] is the extracted audio file
        extracted_path = result[1]
        print(f"Extraction complete: {extracted_path}")

        # Read the result file
        with open(extracted_path, 'rb') as f:
            audio_data = f.read()

        # Cleanup temp files
        os.unlink(noisy_path)
        os.unlink(ref_path)

        # Return the audio file
        return send_file(
            io.BytesIO(audio_data),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='enhanced.wav'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Initialize on startup
init_client()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting VoiceFilter Cloud Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
