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


from .audio_loader import load_audio, load_audio_tensor, get_audio_duration, get_librispeech_files
from .download_data import download_librispeech_sample, AudioDataset

__all__ = [
    'load_audio', 
    'get_audio_duration', 
    'download_librispeech_sample',
    'get_librispeech_files',
    'AudioDataset',
    'DATA_DIR'
]

def can_load_dataset() -> bool:
    """Check if the dataset is already downloaded and accessible."""
    try:
        files = get_librispeech_files()
        return len(files) > 0
    except Exception as e:
        print(f"Dataset check failed: {e}")
        return False
    
def load_dataset() -> AudioDataset:
    """Load the dataset into an AudioDataset object."""
    if not can_load_dataset():
        print("Dataset not found. Attempting to download...")
        download_librispeech_sample() 

    sample_paths = get_librispeech_files()
    print(f"Found {len(sample_paths)} audio files.")
    if len(sample_paths) > 0:
        print(f"Sample: {sample_paths[0]}")
    else:
        print("WARNING: No audio files found. Check download.")
 
 
    return AudioDataset(sample_paths)
