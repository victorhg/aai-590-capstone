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
    
    @staticmethod
    def _apply_perturbation(audio_tensor, perturbation):
        """
        Apply perturbation to audio and clamp to valid range.
        
        Args:
            audio_tensor: Clean audio
            perturbation: Perturbation to add
            
        Returns:
            Perturbed audio clamped to [-1.0, 1.0]
        """
        return torch.clamp(audio_tensor + perturbation, -1.0, 1.0)
    
    @staticmethod
    def calculate_fooling_rate(fooled_count, total_count):
        """
        Calculate fooling rate with zero-division protection.
        
        Args:
            fooled_count: Number of samples fooled
            total_count: Total number of samples
            
        Returns:
            Fooling rate (0.0 to 1.0)
        """
        return fooled_count / total_count if total_count > 0 else 0.0
    
    def _prepare_audio_tensor(self, audio_tensor, target_length):
        """
        Prepare audio tensor: ensure correct shape, device, and length.
        
        Args:
            audio_tensor: Input audio tensor
            target_length: Target audio length in samples
            
        Returns:
            Prepared audio tensor (1, target_length)
        """
        # Ensure correct shape
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_tensor = audio_tensor.to(self.device)
        
        # Pad or crop to target length
        if audio_tensor.shape[1] < target_length:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, target_length - audio_tensor.shape[1])
            )
        else:
            audio_tensor = audio_tensor[:, :target_length]
        
        return audio_tensor
    
    def generate(self, dataset, audio_length=480000, target_fooling_rate=0.8):
        """
        Train universal perturbation on a dataset.
        
        Args:
            dataset: List of audio file paths or AudioDataset
            audio_length: Fixed audio length (30s @ 16kHz = 480,000)
            target_fooling_rate: Stop when fooling rate exceeds this threshold
            
        Returns:
            v: Universal perturbation tensor of shape (1, audio_length)
            history: Dictionary with training metrics
        """
        print(f"Training UAP on {len(dataset)} samples...")
        print(f"Target fooling rate: {target_fooling_rate:.1%}")
        print(f"Epsilon: {self.epsilon}, Xi: {self.xi}, Max Iterations: {self.max_iter}")
        
        # Initialize universal perturbation
        v = torch.zeros(1, audio_length).to(self.device)
        
        # Training history
        history = {
            'fooling_rates': [],
            'losses': [],
            'iterations': []
        }
        
        for iteration in range(self.max_iter):
            print(f"\n=== Iteration {iteration + 1}/{self.max_iter} ===")
            
            fooled_count = 0
            total_count = 0
            epoch_losses = []
            
            # Iterate through dataset
            for idx, audio_path in enumerate(tqdm(dataset, desc="Processing samples")):
                # Load and preprocess audio
                audio_tensor = self._load_audio(audio_path, audio_length)
                if audio_tensor is None:
                    continue
                
                total_count += 1
                
                # Evaluate sample (get original, adversarial, and fooled status)
                result = self._evaluate_single_sample(audio_tensor, v)
                
                # Check if already fooled
                if result['is_fooled']:
                    fooled_count += 1
                    continue
                
                # Compute minimal perturbation for this sample
                delta_i, loss = self._compute_minimal_perturbation(audio_tensor, v)
                epoch_losses.append(loss)
                
                # Update universal perturbation
                v = v + self.xi * delta_i
                v = torch.clamp(v, -self.epsilon, self.epsilon)
            
            # Calculate fooling rate
            fooling_rate = self.calculate_fooling_rate(fooled_count, total_count)
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            
            print(f"Fooling Rate: {fooling_rate:.2%} ({fooled_count}/{total_count})")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Perturbation L∞ Norm: {torch.max(torch.abs(v)):.4f}")
            
            # Store history
            history['fooling_rates'].append(fooling_rate)
            history['losses'].append(avg_loss)
            history['iterations'].append(iteration + 1)
            
            # Check if target fooling rate achieved
            if fooling_rate >= target_fooling_rate:
                print(f"\n✓ Target fooling rate {target_fooling_rate:.1%} achieved!")
                break
        
        print(f"\n=== Training Complete ===")
        print(f"Final Fooling Rate: {history['fooling_rates'][-1]:.2%}")
        print(f"Final Perturbation Norm: {torch.max(torch.abs(v)):.4f}")
        
        return v, history
    
    def _load_audio(self, audio_path, target_length):
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file or tensor
            target_length: Target audio length in samples
            
        Returns:
            audio_tensor: Preprocessed audio tensor (1, target_length)
        """
        try:
            # Import inside method to avoid circular imports
            from src.data.audio_loader import load_audio_tensor
            
            if isinstance(audio_path, (str, Path)):
                _, audio_tensor = load_audio_tensor(str(audio_path))
            else:
                audio_tensor = audio_path
            
            # Prepare audio tensor (shape, device, padding)
            audio_tensor = self._prepare_audio_tensor(audio_tensor, target_length)
            
            return audio_tensor
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def _compute_minimal_perturbation(self, audio_tensor, v):
        """
        Compute minimal perturbation to fool the model for this sample.
        Uses gradient-based approach similar to DeepFool.
        
        Args:
            audio_tensor: Clean audio tensor
            v: Current universal perturbation
            
        Returns:
            delta: Minimal perturbation for this sample
            loss: Loss value (for monitoring)
        """
        audio_tensor = audio_tensor.clone().detach()
        audio_tensor.requires_grad = True
        
        # Apply current perturbation
        perturbed = torch.clamp(audio_tensor + v, -1.0, 1.0)
        
        # Compute loss using the model's attack loss
        loss = self.model.get_loss_for_attack(perturbed)
        
        # Backward pass
        loss.backward()
        
        # Get gradient
        grad = audio_tensor.grad.data
        
        # Compute perturbation direction
        delta = self.epsilon * torch.sign(grad)
        
        # Ensure it's within epsilon ball
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        
        return delta, loss.item()
    
    def evaluate(self, dataset, v, audio_length=480000):
        """
        Evaluate the universal perturbation on a dataset.
        
        Args:
            dataset: List of audio file paths
            v: Universal perturbation tensor
            audio_length: Fixed audio length
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        print(f"\nEvaluating UAP on {len(dataset)} samples...")
        
        fooled_count = 0
        total_count = 0
        results_list = []
        
        for audio_path in tqdm(dataset, desc="Evaluating"):
            audio_tensor = self._load_audio(audio_path, audio_length)
            if audio_tensor is None:
                continue
            
            total_count += 1
            
            # Evaluate sample
            result = self._evaluate_single_sample(audio_tensor, v)
            
            # Count fooled samples
            if result['is_fooled']:
                fooled_count += 1
            
            results_list.append({
                'audio_path': str(audio_path),
                'original': result['original'],
                'adversarial': result['adversarial'],
                'fooled': result['is_fooled']
            })
        
        fooling_rate = self.calculate_fooling_rate(fooled_count, total_count)
        
        # Compute SNR (note: UAP SNR is measured against assumed typical audio RMS)
        # For accurate SNR, pass clean audio samples to compute_snr_universal()
        snr = self._compute_snr_universal(v)
        
        results = {
            'fooling_rate': fooling_rate,
            'fooled_count': fooled_count,
            'total_count': total_count,
            'snr_db': snr,
            'perturbation_linf': torch.max(torch.abs(v)).item(),
            'samples': results_list
        }
        
        print(f"\n=== Evaluation Results ===")
        print(f"Fooling Rate: {fooling_rate:.2%} ({fooled_count}/{total_count})")
        print(f"SNR: {snr:.2f} dB")
        print(f"L∞ Norm: {results['perturbation_linf']:.4f}")
        
        return results
    
    @staticmethod
    def compute_snr(original, perturbed):
        """
        Compute Signal-to-Noise Ratio (SNR) in dB.
        Standardized version matching pgd.py signature.
        
        Args:
            original: Clean audio (numpy array or torch tensor)
            perturbed: Adversarial audio (numpy array or torch tensor)
            
        Returns:
            SNR in dB
        """
        # Convert to numpy if needed
        if torch.is_tensor(original):
            original = original.detach().cpu().numpy()
        if torch.is_tensor(perturbed):
            perturbed = perturbed.detach().cpu().numpy()
        
        original = original.astype(np.float32)
        perturbed = perturbed.astype(np.float32)
        
        noise = perturbed - original
        
        signal_power = np.sum(original ** 2)
        noise_power = np.sum(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    def _compute_snr_universal_universal(self, perturbation):
        """
        Compute Signal-to-Noise Ratio for universal perturbation.
        Uses assumed typical audio RMS for signal power estimation.
        
        Note: For accurate SNR on specific audio, use compute_snr(original, perturbed).
        
        Args:
            perturbation: Universal perturbation tensor
            
        Returns:
            snr_db: SNR in decibels
        """
        # For universal perturbation, we measure its power relative to typical audio
        # Assuming typical audio has RMS around 0.1 (normalized audio)
        signal_power = 0.1 ** 2
        noise_power = torch.mean(perturbation ** 2).item()
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def _to_device(self, tensor):
        """Transfer tensor to model's device."""
        return tensor.to(self.device)
    
    def save_perturbation(self, v, filepath):
        """Save universal perturbation to file."""
        torch.save(v, filepath)
        print(f"Perturbation saved to {filepath}")
    
    def load_perturbation(self, filepath):
        """Load universal perturbation from file."""
        v = torch.load(filepath, map_location=self.device)
        print(f"Perturbation loaded from {filepath}")
        return v
