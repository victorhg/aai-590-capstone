"""
Shared utilities for adversarial attack implementations.
"""
import torch
import numpy as np
import jiwer

_NORMALIZER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def normalize_text(text: str) -> str:
    return _NORMALIZER(text)


def tile_to_length(delta: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Crop or tile ``delta`` to match ``target_len`` samples (AGENTS.md §6).

    Args:
        delta:      Perturbation tensor of shape ``(1, uap_len)``.
        target_len: Desired number of samples.

    Returns:
        Tensor of shape ``(1, target_len)``.
    """
    uap_len = delta.shape[-1]
    if target_len < uap_len:
        return delta[:, :target_len]
    if target_len > uap_len:
        repeats = (target_len + uap_len - 1) // uap_len
        return delta.repeat(1, repeats)[:, :target_len]
    return delta


def prepare_audio(audio: torch.Tensor, device: str) -> torch.Tensor:
    """
    Ensure ``audio`` is 2-D ``(1, T)`` and moved to ``device``.

    Args:
        audio:  1-D or 2-D audio waveform tensor.
        device: Target device string (``'cpu'``, ``'cuda'``, ``'mps'``).

    Returns:
        2-D float32 tensor on the requested device.
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    return audio.to(device)


def compute_snr(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        original: Clean audio (numpy array).
        perturbed: Adversarial audio (numpy array).
        
    Returns:
        SNR in dB.
    """
    original = original.astype(np.float32)
    perturbed = perturbed.astype(np.float32)
    
    noise = perturbed - original
    
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)