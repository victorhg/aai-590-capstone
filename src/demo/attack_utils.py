"""
Utility functions for applying adversarial perturbations in real-time demos.
"""

import torch
import numpy as np
from typing import Optional

class UAPManager:
    def __init__(self, uap_path: str, epsilon: float = 0.01):
        """
        Initialize the Universal Adversarial Perturbation (UAP) manager.
        
        Args:
            uap_path: Path to the saved .pt file containing the UAP vector.
            epsilon: The magnitude of the perturbation (L_inf norm constraint).
        """
        self.uap_path = uap_path
        self.epsilon = epsilon
        self.uap_vector = None
        self.load_uap()

    def load_uap(self):
        """Load the pre-trained UAP vector."""
        try:
            self.uap_vector = torch.load(self.uap_path)
            print(f"Successfully loaded UAP from {self.uap_path}. Shape: {self.uap_vector.shape}")
        except FileNotFoundError:
            print(f"Error: UAP file not found at {self.uap_path}. Please train or provide the model.")
            self.uap_vector = None
        except Exception as e:
            print(f"Error loading UAP: {e}")
            self.uap_vector = None

    def apply_uap(self, audio_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Apply the UAP to the input audio tensor.
        
        Args:
            audio_data: Input audio tensor (shape: [channels, samples] or [samples]).
            
        Returns:
            Perturbed audio tensor or None if loading failed.
        """
        if self.uap_vector is None:
            return None

        # Ensure input is a tensor and on the same device
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data)
            
        audio_data = audio_data.to(self.uap_vector.device)
        
        # UAP logic:
        # 1. Handle mismatched lengths: UAP is typically fixed length (e.g., 30s).
        #    If audio is shorter, crop UAP. If longer, tile UAP.
        
        uap_len = self.uap_vector.shape[-1]
        audio_len = audio_data.shape[-1]
        
        # Add channel dimension if missing (assuming mono input, but checking for robustness)
        if len(audio_data.shape) == 1:
            audio_data = audio_data.unsqueeze(0)
            
        # Slice or Tile UAP to match audio length
        if audio_len >= uap_len:
            # Tile: Repeat the UAP vector
            tiles = (audio_len + uap_len - 1) // uap_len
            # Take only the first 'tiles' tiles to match exact length
            required_uap = self.uap_vector.repeat(1, tiles)[:, :audio_len]
        else:
            # Crop: Take first N seconds of UAP
            required_uap = self.uap_vector[:, :audio_len]
            
        # Normalize UAP to have same variance as audio to prevent clipping artifacts
        # (Optional, but good practice for seamless addition)
        uap_std = required_uap.std()
        audio_std = audio_data.std()
        if uap_std > 0 and audio_std > 0:
            scale = audio_std / uap_std
            required_uap = required_uap * scale

        # Apply Perturbation
        perturbed_audio = audio_data + required_uap
        
        # Apply L_inf epsilon clipping to enforce constraint
        perturbed_audio = torch.clamp(perturbed_audio, -1.0, 1.0)
        
        return perturbed_audio

    def get_snr(self, original: torch.Tensor, perturbed: torch.Tensor) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB.
        """
        if original.shape != perturbed.shape:
            return 0.0
            
        signal_power = torch.mean(original ** 2).item()
        noise_power = torch.mean((original - perturbed) ** 2).item()
        
        if noise_power == 0:
            return 100.0 # Infinite SNR
            
        return 10 * np.log10(signal_power / noise_power)
