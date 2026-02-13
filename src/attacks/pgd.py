"""
PGD Attack Implementation for Whisper ASR.

This module implements the Projected Gradient Descent (PGD) attack 
on Whisper models. It computes gradients with respect to the input audio
to generate adversarial perturbations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import librosa


class PGDAttack:
    """
    Projected Gradient Descent (PGD) Attack for Whisper.
    """
    
    def __init__(
        self, 
        model, 
        epsilon: float = 0.01, 
        alpha: float = 0.003, 
        num_iter: int = 10, 
        attack_type: str = "untargeted",
        random_start: bool = True
    ):
        """
        Args:
            model: Whisper model instance.
            epsilon: Max perturbation magnitude (L_inf norm).
            alpha: Step size for PGD.
            num_iter: Number of optimization steps.
            attack_type: 'untargeted' or 'targeted'.
            random_start: Whether to initialize with random noise.
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.attack_type = attack_type
        self.random_start = random_start
        
        # Ensure model is in eval mode for attack
        self.model.eval()

    def _verify_input_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Ensure input audio is at 16kHz and normalized to [-1, 1].
        
        Args:
            audio: Input tensor (1, samples) or (samples).
            
        Returns:
            Tensor shaped (1, samples) with float32 range [-1, 1].
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Normalize to [-1, 1]
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio

    def generate(
        self, 
        audio: torch.Tensor, 
        input_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate adversarial perturbation.
        
        Args:
            audio: Clean audio tensor (1, samples).
            input_lengths: Optional tensor of actual sample counts per batch.
            
        Returns:
            Adversarial audio tensor.
        """
        # Deep copy to avoid modifying original
        adv_audio = audio.clone().detach()
        
        if self.random_start:
            # Random start within epsilon-ball
            start_delta = torch.empty_like(adv_audio).uniform_(
                -self.epsilon, self.epsilon
            )
            adv_audio = torch.clamp(adv_audio + start_delta, -1.0, 1.0)

        for _ in range(self.num_iter):
            adv_audio.requires_grad = True
            
            # Forward pass
            with torch.enable_grad():
                # Check if model has a dedicated loss function for attacks
                if hasattr(self.model, "get_loss_for_attack"):
                    loss = self.model.get_loss_for_attack(adv_audio)
                else:
                    # Generic fall-back for transformers models
                    output = self.model(adv_audio)
                    logits = output.logits
                    # Simple untargeted: Maximize uncertainty (entropy) of first token
                    # or just minimize max logit
                    probs = F.softmax(logits[:, 0, :], dim=-1)
                    loss = -torch.max(probs, dim=-1)[0].mean()
                
            # Compute gradient
            grad = torch.autograd.grad(loss, adv_audio, retain_graph=False)[0]
            
            # Update (Sign of gradient)
            if self.attack_type == "untargeted":
                # We want to maximize the loss (maximize uncertainty)
                # So we move in the direction of the gradient
                adv_audio = adv_audio + self.alpha * torch.sign(grad)
            elif self.attack_type == "targeted":
                # We want to minimize the loss (minimize distance to target)
                # So we move against the gradient
                adv_audio = adv_audio - self.alpha * torch.sign(grad)

            # Project back to epsilon-ball
            adv_audio = torch.clamp(adv_audio, audio - self.epsilon, audio + self.epsilon)
            adv_audio = torch.clamp(adv_audio, -1.0, 1.0) # Final clamp to audio domain
            
            # Detach for next iteration
            adv_audio = adv_audio.detach()

        return adv_audio

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

