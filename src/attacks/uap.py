"""
Universal Adversarial Perturbation (UAP) Attack
Based on Moosavi-Dezfooli et al. (2017)

Generates a single perturbation that fools the model on multiple samples.
"""

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

import src.data as data_loader

class UAPDataset(Dataset):
    """Loads audio files and pads/crops them to a fixed length for UAP training."""

    def __init__(self, files, max_duration=5.0, sample_rate=16000):
        self.files = files
        self.max_samples = int(max_duration * sample_rate)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        _, audio = data_loader.load_audio_tensor(self.files[idx])
        return data_loader.pad_or_crop_audio(audio, self.max_samples)


class UniversalPerturbation:
    """
    Trains a single perturbation v that degrades Whisper's transcription
    across many audio samples (Moosavi-Dezfooli et al. 2017).

    Usage:
        attack = UniversalPerturbation(model, epsilon=0.05)
        v, history = attack.generate(dataset, epochs=20)
    """
    
    def __init__(self, model, epsilon=0.02, lr=0.01, ):
        """
        Initialize UAP attack.
        
        Args:
            model: WhisperASRWithAttack wrapper
            epsilon: L_inf constraint on perturbation magnitude
            lr: Learning rate for v optimizing the perturbation
        """
        self.model = model
        self.epsilon = epsilon
        self.lr = lr
        self.device = next(model.parameters()).device
    
    
    def _apply_perturbation(self, audio_tensor, perturbation):
        return torch.clamp(audio_tensor + perturbation, -1.0, 1.0)
    
    def generate(self, dataset, audio_length=5.0, epochs=10):

        history = {
            'fooling_rates': [],
            'avg_losses': []
        }
       
        # initialize universal perturbation v
        uap_length = int(audio_length * 16000)
        v = torch.zeros(1, uap_length, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([v], lr=self.lr) 

        print(f"UAP Length: {uap_length} samples ({audio_length}s)")
        
        for epoch in range(epochs):
            
            epoch_loss = 0.0
            fooled_count = 0
        
            for audio in tqdm(dataset, desc=f"Epoch({epoch+1}/{epochs}) - Processing samples"):
                
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
                audio = audio.to(self.device)

                # v already fools the model on this sample?
                result = self.evaluate_single_sample(audio, v)
                if result['is_fooled']:
                    fooled_count += 1
                    continue  # No need to update v for this sample
                
                x_adv = self._apply_perturbation(audio, v)
                loss = self.model.get_loss_for_attack(x_adv)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Project v back to epsilon ball (L_inf constraint)
                with torch.no_grad():
                    v.clamp_(-self.epsilon, self.epsilon)
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataset)
            fooling_rate = fooled_count / len(dataset)
            
            
            # Store history
            history['fooling_rates'].append(fooling_rate)
            history['avg_losses'].append(avg_loss)
            
        return v.detach(), history

    
    def save_perturbation(self, v, filepath):
        torch.save(v, filepath)
    
    def load_perturbation(self, filepath):
        v = torch.load(filepath, map_location=self.device)
        return v

    def evaluate_single_sample(self, audio_tensor, perturbation):
        """
        Evaluate whether a perturbation fools the model on a single sample.
        """
        
        with torch.no_grad():
            orig_transcription = self.model.transcribe(audio_tensor)
            adv_transcription = self.model.transcribe(self._apply_perturbation(audio_tensor, perturbation))
            
        is_fooled = (orig_transcription != adv_transcription)
        
        return {
            'original': orig_transcription,
            'adversarial': adv_transcription,
            'is_fooled': is_fooled
        }
    
    def apply_uap_to_audio(self, audio_tensor, uap_vector, target_length=None):
        """Tile uap_vector to match audio length, add it, and clamp to [-1, 1]."""
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if target_length is None:
            target_length = audio_tensor.shape[-1]

        uap_length = uap_vector.shape[-1]
        num_tiles = (target_length + uap_length - 1) // uap_length
        uap_tiled = uap_vector.repeat(1, num_tiles)[:, :target_length]

        audio_tensor = data_loader.pad_or_crop_audio(audio_tensor, target_length)
        return torch.clamp(audio_tensor + uap_tiled.to(audio_tensor.device), -1.0, 1.0)
    



    