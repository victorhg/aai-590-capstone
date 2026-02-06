"""
Carlini-Wagner Attack Implementation for Whisper Audio ASR.
Implementation of the targeted L2 attack.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from transformers import WhisperProcessor
import numpy as np

# Assuming imports from the main project structure
try:
    from src.attacks.pgd import PGDAttack
    HAS_PGD = True
except ImportError:
    HAS_PGD = False

class CarliniWagnerAttack:
    """
    Targeted Adversarial Attack using the Carlini-Wagner formulation (L2 norm).
    This attack optimizes for a specific target text while minimizing perturbation magnitude.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_text: str, 
        epsilon: float = 0.1, 
        learning_rate: float = 0.01, 
        iterations: int = 100, 
        alpha: float = 0.01,
        device: Optional[str] = None
    ):
        """
        Args:
            model: The Whisper model instance.
            target_text: The specific text to be output by the model.
            epsilon: Maximum perturbation magnitude (L2 norm).
            learning_rate: Step size for optimizer.
            iterations: Number of optimization steps.
            alpha: Constant multiplier for the loss (margin).
            device: Device to run on (cuda/cpu).
        """
        self.model = model
        self.target_text = target_text
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.alpha = alpha
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model.to(self.device)
        self.model.eval()
        
        # Store original model state
        self.model.requires_grad_(False)
        self.model.eval()

    def _tokenize_target(self, processor):
        """Convert target text to input ids."""
        return processor.tokenizer(
            self.target_text, 
            return_tensors="pt", 
            add_special_tokens=False
        )["input_ids"].to(self.device)

    def attack(self, audio_tensor: torch.Tensor, processor) -> torch.Tensor:
        """
        Perform the CW attack.
        
        Args:
            audio_tensor: Clean audio tensor (un-normalized).
            processor: Whisper processor for preprocessing/audio-to-text.
            
        Returns:
            perturbed_audio: The adversarial audio tensor.
        """
        # 1. Initialize perturbation with random noise
        delta = torch.randn_like(audio_tensor, requires_grad=True)
        
        # Move perturbation to target device
        delta = delta.to(self.device)
        
        # Target Input IDs
        target_ids = self._tokenize_target(processor)
        
        # Optimizer
        optimizer = optim.Adam([delta], lr=self.learning_rate)
        
        # Pre-process input (Mel spectrogram)
        # Note: Whisper preprocesses audio internally in the forward pass.
        # We ensure the input tensor is on the correct device and type.
        
        with torch.set_grad_enabled(True):
            self.model.train() # Enable gradient tracking for the perturbation
            
            for step in range(self.iterations):
                # Zero grad
                optimizer.zero_grad()
                
                # Create adversarial input
                adv_input = audio_tensor + delta
                
                # Ensure input is within valid bounds [-1, 1] for Whisper
                adv_input = torch.clamp(adv_input, -1.0, 1.0)
                
                # Forward pass
                # Whisper requires (batch_size, seq_len)
                # We add a batch dim if not present
                if adv_input.dim() == 1:
                    adv_input = adv_input.unsqueeze(0)
                    
                try:
                    # Run Whisper
                    outputs = self.model(inputs=adv_input, return_dict=True)
                    
                    # We want to maximize the probability of the TARGET tokens.
                    # Standard CrossEntropy loss does this (minimizing negative log prob).
                    # We mask loss to only apply to the target tokens if possible, 
                    # or apply full sequence loss for simplicity in this implementation.
                    
                    # Get logits for decoder
                    # shape: (batch, seq_len, vocab_size)
                    logits = outputs.get("decoder_output", outputs.logits) 
                    
                    # Compute loss on target tokens only
                    # Note: This is a simplified "Targeted" logic. 
                    # True CW involves maximizing (c - f(x)) where f(x) is loss for non-targets.
                    # Here we focus on minimizing loss for target.
                    
                    # Reshape for loss calculation
                    loss_fn = nn.CrossEntropyLoss(reduction='mean')
                    
                    # Flatten target tokens across batch and time
                    # We calculate loss against the target sequence.
                    # This is a heuristic for "Targeted CW" within a specific codebase context.
                    loss = loss_fn(logits.permute(0, 2, 1), target_ids.repeat(logits.size(0), 1))
                    
                except Exception as e:
                    print(f"Error in forward pass step {step}: {e}")
                    continue

                # Backward pass
                loss.backward()
                
                # Update delta
                optimizer.step()
                
                # L2 Projection Step (The core of CW)
                # We need to ensure ||delta||_2 <= epsilon
                with torch.no_grad():
                    norm = torch.norm(delta)
                    if norm > self.epsilon:
                        delta.data = delta.data * (self.epsilon / norm)
                    
                    # Clamp audio bounds again
                    delta.data = torch.clamp(delta, -1.0, 1.0)

        return (audio_tensor + delta).detach().cpu()

    def batch_attack(self, audio_batch: torch.Tensor, processor) -> torch.Tensor:
        """Wrapper to handle batch processing."""
        # For CW, batch processing is complex due to variable target tokens.
        # We iterate for now to preserve accuracy.
        adv_batch = []
        for audio in audio_batch:
            adv = self.attack(audio.unsqueeze(0), processor)
            adv_batch.append(adv)
        return torch.cat(adv_batch)
