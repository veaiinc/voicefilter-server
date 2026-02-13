"""
VoiceFilter API - Hugging Face Spaces Deployment
Simple Gradio app with API access.
"""

import os
import tempfile
import gradio as gr
from gradio_client import Client, handle_file

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

        result = hf_tse_client.predict(
            handle_file(noisy_audio),
            handle_file(reference_audio),
            "mix",
            "iter_model",
            api_name="/gradio_TSE"
        )

        extracted_path = result[1]
        print(f"Extraction complete: {extracted_path}")
        return extracted_path

    except Exception as e:
        print(f"Error processing audio: {e}")
        raise gr.Error(f"Processing failed: {str(e)}")

# Initialize client on startup
init_client()

# Create Gradio interface with API enabled
demo = gr.Interface(
    fn=filter_voice,
    inputs=[
        gr.Audio(label="Noisy/Mixed Audio", type="filepath"),
        gr.Audio(label="Reference Audio (Target Speaker)", type="filepath")
    ],
    outputs=gr.Audio(label="Extracted Voice", type="filepath"),
    title="VoiceFilter - Speaker Voice Isolation",
    description="Extract a target speaker's voice from noisy/mixed audio using a reference sample of their voice.",
    allow_flagging="never",
    api_name="filter"
)

# Launch
demo.launch()
