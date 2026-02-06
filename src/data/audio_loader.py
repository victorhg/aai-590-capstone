import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Union

def load_audio(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio file and normalize it for Whisper.
    
    Steps:
    1. Load audio using librosa (resampling handled by librosa or sf).
    2. Convert to float32.
    3. Normalize to [-1.0, 1.0] range.
    
    Args:
        file_path: Path to the audio file.
        target_sr: Target sample rate (Whisper default: 16000).
        
    Returns:
        torch.Tensor: Normalized audio tensor of shape (samples,).
    """
    try:
        # Load audio. librosa automatically resamples if 'sr' is provided
        y, sr = librosa.load(file_path, sr=target_sr)
        
        # Normalize to [-1.0, 1.0]
        # librosa.load already returns normalized float, but we enforce it
        y = np.clip(y, -1.0, 1.0)
        
        # Convert to torch tensor
        # Whisper expects float32 input
        audio_tensor = torch.from_numpy(y).float()
        
        return audio_tensor
        
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {file_path}: {e}")

def get_audio_duration(file_path: str) -> float:
    """
    Get duration of an audio file in seconds without loading it.
    """
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception as e:
        raise RuntimeError(f"Error getting audio info for {file_path}: {e}")

if __name__ == "__main__":
    # Test loading
    print("Audio loader module loaded successfully.")
