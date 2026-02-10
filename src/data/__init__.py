"""
Data module for ASR Adversarial Attacks.
Handles loading, preprocessing, and downloading datasets.
"""
import os
from pathlib import Path

# Data Paths
# Resolves to: /Users/.../soundfinal/src/data/../../data -> /Users/.../soundfinal/data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

TARGET_SAMPLE_RATE = 16000  # Whisper's expected sample rate


from .audio_loader import load_audio, get_audio_duration, get_librispeech_files
from .download_data import download_librispeech_sample, AudioDataset

__all__ = [
    'load_audio', 
    'get_audio_duration', 
    'download_librispeech_sample',
    'get_librispeech_files',
    'AudioDataset',
    'DATA_DIR'
]
