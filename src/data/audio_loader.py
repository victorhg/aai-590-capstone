"""
Audio loading and preprocessing utilities.
Ensures compatibility with Whisper (16kHz, float32, [-1, 1]).
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional
import os

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """
    Load an audio file and ensure it's in the correct format.
    
    Args:
        path: Path to the audio file.
        sr: Target sampling rate (default 16000 for Whisper).
    
    Returns:
        Audio waveform as a numpy array of dtype float32.
    """
    # Load audio file
    y, orig_sr = librosa.load(path, sr=sr)
    return y.astype(np.float32)

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to a target sampling rate.
    
    Args:
        audio: Input audio waveform.
        orig_sr: Original sampling rate.
        target_sr: Target sampling rate.
    
    Returns:
        Resampled audio waveform.
    """
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to the range [-1, 1].
    
    Args:
        audio: Input audio waveform (assumed to be in the correct range or PCM).
    
    Returns:
        Normalized audio waveform.
    """
    # librosa.load usually handles PCM to [-1, 1] conversion,
    # but we enforce it here to be safe.
    return np.clip(audio / np.max(np.abs(audio)), -1.0, 1.0)

def get_audio_info(path: str) -> Tuple[int, int]:
    """
    Get duration and sample rate of an audio file.
    
    Args:
        path: Path to the audio file.
    
    Returns:
        Tuple of (duration in seconds, sample rate).
    """
    y, sr = librosa.load(path, sr=None)
    return len(y) / sr, sr

def save_audio(path: str, audio: np.ndarray, sr: int = 16000):
    """
    Save an audio file.
    
    Args:
        path: Output path.
        audio: Audio waveform.
        sr: Sample rate.
    """
    sf.write(path, audio, sr)
