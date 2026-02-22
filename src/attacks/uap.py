"""
Universal Adversarial Perturbation (UAP) Attack
Based on Moosavi-Dezfooli et al. (2017)

Generates a single perturbation that fools the model on multiple samples.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import Dataset

import src.data as data_loader

class UAPDataset(Dataset):
    """
    Dataset for UAP training that preprocesses all audio to a fixed length.
    
    This ensures consistent tensor shapes for training the universal perturbation.
    All audio samples are padded (with zeros) or cropped to match max_duration.
    
    Args:
        files: List of audio file paths
        max_duration: Target audio length in seconds (should match UAP training length)
        sample_rate: Audio sample rate in Hz (default: 16000 for Whisper)
    """
    def __init__(self, files, max_duration=5.0, sample_rate=16000):
        self.files = files
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load and preprocess audio to fixed length.
        
        Returns:
            audio_tensor: 1D tensor of shape (max_samples,)
        """
        path = self.files[idx]
        _, audio_tensor = data_loader.load_audio_tensor(path)
        
        # Pad or crop to fixed length
        return data_loader.pad_or_crop_audio(audio_tensor, self.max_samples)

class UniversalPerturbation:
    """
    Universal Adversarial Perturbation following Moosavi-Dezfooli et al. (2017).
    
    Trains a single perturbation vector that works across multiple audio samples.
    """
    
    def __init__(self, model, epsilon=0.02, max_iter=10, xi=0.01, batch_size=1):
        """
        Initialize UAP attack.
        
        Args:
            model: WhisperASRWithAttack wrapper
            epsilon: L_inf constraint on perturbation magnitude
            max_iter: Maximum iterations over the dataset
            xi: Overshoot parameter for updating universal perturbation
            batch_size: Number of samples per iteration (default 1 for memory)
        """
        self.model = model
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.xi = xi
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
    
    def _evaluate_single_sample(self, audio_tensor, perturbation):
        """
        Evaluate whether a perturbation fools the model on a single sample.
        
        Args:
            audio_tensor: Clean audio tensor
            perturbation: Perturbation to apply
            
        Returns:
            dict with keys 'original', 'adversarial', 'is_fooled'
        """
        # Get original transcription
        with torch.no_grad():
            orig_transcription = self.model.transcribe(audio_tensor)
        
        # Apply perturbation
        perturbed = self._apply_perturbation(audio_tensor, perturbation)
        
        # Get adversarial transcription
        with torch.no_grad():
            adv_transcription = self.model.transcribe(perturbed)
        
        # Check if fooled
        is_fooled = (orig_transcription != adv_transcription)
        
        return {
            'original': orig_transcription,
            'adversarial': adv_transcription,
            'is_fooled': is_fooled
        }
    
    
    def _apply_perturbation(self, audio_tensor, perturbation):
        return torch.clamp(audio_tensor + perturbation, -1.0, 1.0)
    
    def initialize_perturbation(self, audio_length):
        return torch.zeros(1, audio_length, device=self.device, requires_grad=True)
    
    def generate(self, dataset, audio_length=5.0, 
                 lr=0.01, epochs=10):

        history = {
            'fooling_rates': [],
            'avg_losses': [],
            'epochs': []
        }
       
        uap_length = int(audio_length * 16000)
        v = self.initialize_perturbation(uap_length)
        print(f"UAP Length: {uap_length} samples ({audio_length}s)")
        
        # Optimizer for v
        optimizer = torch.optim.SGD([v], lr=lr)
        
        for epoch in range(epochs):
            
            epoch_loss = 0.0
            num_samples = 0
            fooled_count = 0
        
            for audio in tqdm(dataset, desc=f"Epoch({epoch+1}/{epochs}) - Processing samples"):
                
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
                audio = audio.to(self.device)

                result = self._evaluate_single_sample(audio, v)
                if result['is_fooled']:
                    fooled_count += 1
                
                x_adv = self._apply_perturbation(audio, v)

                loss = self.model.get_loss_for_attack(x_adv)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Project v back to epsilon ball (L_inf constraint)
                with torch.no_grad():
                    #v.data += self.xi * delta
                    v.clamp_(-self.epsilon, self.epsilon)
                
                epoch_loss += loss.item()
                num_samples += 1
            
            avg_loss = epoch_loss / num_samples
            fooling_rate = fooled_count / num_samples
            v_norm = v.norm().item()
            v_max = v.abs().max().item()
            
            # Store history
            history['fooling_rates'].append(fooling_rate)
            history['avg_losses'].append(avg_loss)
            history['epochs'].append(epoch+1)
            
        return v.detach(), history

    
 
    def save_perturbation(self, v, filepath):
        torch.save(v, filepath)
    
    def load_perturbation(self, filepath):
        v = torch.load(filepath, map_location=self.device)
        return v
    



    