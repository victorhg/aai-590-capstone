"""
Data module for ASR Adversarial Attacks.
Handles loading, preprocessing, and downloading datasets.
"""

from .audio_loader import load_audio, resample_audio, normalize_audio

__all__ = ['load_audio', 'resample_audio', 'normalize_audio']
