import torch
import numpy as np
from pathlib import Path

# Paths to existing UAP results
UAP_PATH = Path("results/universal_perturbation_v.pt")
EPSILON = 0.03  # L-infinity constraint

def load_uap_vector(path=UAP_PATH):
    """
    Loads the universal adversarial perturbation vector.
    Returns a tensor of shape (channels, samples).
    """
    if not path.exists():
        raise FileNotFoundError(f"UAP vector not found at {path}. Please run 04_uap_training.ipynb first.")
    
    uap_data = torch.load(path, map_location='cpu')
    
    # Handle different tensor shapes (1D vs 2D)
    if uap_data.dim() == 1:
        # (samples,) -> (1, samples)
        return uap_data.unsqueeze(0)
    elif uap_data.dim() == 2 and uap_data.shape[0] == 1:
        # (1, samples) -> keep as is
        return uap_data
    else:
        return uap_data

def apply_uap(audio_chunk, uap_vector, epsilon=EPSILON):
    """
    Applies the universal perturbation to an audio chunk.
    
    Args:
        audio_chunk (torch.Tensor): Input audio tensor. Shape (channels, samples).
        uap_vector (torch.Tensor): Pre-trained UAP tensor. Shape (channels, samples) or (samples,).
        epsilon (float): Max perturbation magnitude (L-inf).
    
    Returns:
        torch.Tensor: Perturbed audio tensor, clamped to [-1, 1].
    """
    # Ensure UAP is on the same device as audio
    uap = uap_vector.to(audio_chunk.device)
    
    # Handle Length Mismatch
    # UAP is usually trained on fixed length or tiled. 
    # We will tile if the chunk is longer than the UAP.
    if audio_chunk.shape[1] > uap.shape[1]:
        repeats = int(np.ceil(audio_chunk.shape[1] / uap.shape[1]))
        uap_perturbation = uap.repeat(1, repeats)[:, :audio_chunk.shape[1]]
    else:
        uap_perturbation = uap[:, :audio_chunk.shape[1]]

    # Add perturbation
    perturbed_audio = audio_chunk + uap_perturbation
    
    # Enforce L-inf constraint (clipping)
    perturbed_audio = torch.clamp(perturbed_audio, -1.0, 1.0)
    
    return perturbed_audio
