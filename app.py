"""
VoiceFilter API - Hugging Face Spaces Deployment
Uses Gradio for the UI + FastAPI for REST API access.
"""

import os
import tempfile
import shutil
import gradio as gr
from gradio_client import Client, handle_file
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize HuggingFace client for Target Speaker Extraction
hf_tse_client = None

def init_client():
    """Initialize HuggingFace API client"""
    global hf_tse_client
    try:
        hf_tse_client = Client("swc2/Target-speaker-extraction")
        print("HuggingFace Target Speaker Extraction API connected")
        return True
    except Exception as e:
        print(f"Failed to connect to HuggingFace API: {e}")
        return False

def filter_voice(noisy_audio, reference_audio):
    """
    Extract target speaker's voice from noisy audio using reference.
    """
    global hf_tse_client

    if hf_tse_client is None:
        if not init_client():
            raise gr.Error("Failed to connect to HuggingFace API")

    if noisy_audio is None or reference_audio is None:
        raise gr.Error("Both noisy and reference audio files are required")

    try:
        print(f"Processing: noisy={noisy_audio}, reference={reference_audio}")

        # Call HuggingFace API
        result = hf_tse_client.predict(
            handle_file(noisy_audio),
            handle_file(reference_audio),
            "mix",
            "iter_model",
            api_name="/gradio_TSE"
        )

        # Result[1] is the extracted audio file path
        extracted_path = result[1]
        print(f"Extraction complete: {extracted_path}")

        return extracted_path

    except Exception as e:
        print(f"Error processing audio: {e}")
        raise gr.Error(f"Processing failed: {str(e)}")

# Initialize client on startup
init_client()

# Create Gradio interface
demo = gr.Interface(
    fn=filter_voice,
    inputs=[
        gr.Audio(label="Noisy/Mixed Audio", type="filepath"),
        gr.Audio(label="Reference Audio (Target Speaker)", type="filepath")
    ],
    outputs=gr.Audio(label="Extracted Voice", type="filepath"),
    title="VoiceFilter - Speaker Voice Isolation",
    description="Extract a target speaker's voice from noisy/mixed audio using a reference sample of their voice.",
    examples=[],
    allow_flagging="never"
)

# Get the FastAPI app from Gradio
app = demo.app

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API endpoints for Swift app compatibility

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({
        "status": "ok",
        "hf_tse_available": hf_tse_client is not None,
        "embedder_loaded": hf_tse_client is not None,
        "voicefilter_loaded": hf_tse_client is not None
    })

@app.post("/filter-with-reference")
async def filter_with_reference(
    noisy: UploadFile = File(...),
    reference: UploadFile = File(...)
):
    """
    Extract target speaker using reference audio.
    Compatible with the Swift VoiceFilterService.
    """
    global hf_tse_client

    if hf_tse_client is None:
        if not init_client():
            return JSONResponse({"error": "HuggingFace API not available"}, status_code=500)

    try:
        # Save uploaded files to temp
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            content = await noisy.read()
            tmp.write(content)
            noisy_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            content = await reference.read()
            tmp.write(content)
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

        # Cleanup input temp files
        os.unlink(noisy_path)
        os.unlink(ref_path)

        # Return the audio file
        return FileResponse(
            extracted_path,
            media_type="audio/wav",
            filename="enhanced.wav"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# Launch with API enabled
if __name__ == "__main__":
    demo.launch()
