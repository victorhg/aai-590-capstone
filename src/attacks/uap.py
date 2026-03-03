"""
Universal Adversarial Perturbation (UAP) Attack
Based on Moosavi-Dezfooli et al. (2017)

Generates a single perturbation that fools the model on multiple samples.
"""

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

import src.data as data_loader
from .base import BaseUniversalAttack
from .utils import prepare_audio

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


class UniversalPerturbation(BaseUniversalAttack):
    """
    Trains a single perturbation v that degrades Whisper's transcription
    across many audio samples (Moosavi-Dezfooli et al. 2017).

    Inherits ``apply``, ``get_perturbation``, ``save``, and ``load`` from
    :class:`~attacks.base.BaseUniversalAttack`.

    Usage:
        attack = UniversalPerturbation(model, epsilon=0.05)
        v, history = attack.generate(dataset, epochs=20)
    """
    
    def __init__(self, model, epsilon=0.02, lr=0.01):
        """
        Initialize UAP attack.

        Args:
            model:   WhisperASRWithAttack wrapper.
            epsilon: L_inf constraint on perturbation magnitude.
            lr:      Learning rate for optimizing v.
        """
        self.model   = model
        self.epsilon = epsilon
        self.lr      = lr
        self.device  = next(model.parameters()).device
        # δ is initialised lazily in generate() once audio_length is known.
        self.delta: torch.Tensor | None = None
    
    def generate(self, dataset, audio_length=5.0, epochs=10):

        history = {'fooling_rates': [], 'avg_losses': []}

        # Initialise δ (stored on self so BaseUniversalAttack helpers work).
        uap_length = int(audio_length * 16000)
        self.delta = torch.zeros(1, uap_length, device=self.device, requires_grad=True)
        optimizer  = torch.optim.SGD([self.delta], lr=self.lr)

        print(f"UAP Length: {uap_length} samples ({audio_length}s)")

        for epoch in range(epochs):
            epoch_loss   = 0.0
            fooled_count = 0

            for audio in tqdm(dataset, desc=f"Epoch({epoch+1}/{epochs}) - Processing samples"):
                audio = prepare_audio(audio, self.device)

                # Skip samples the current δ already fools.
                if self.evaluate_single_sample(audio)['is_fooled']:
                    fooled_count += 1
                    continue

                x_adv = self._apply_perturbation(audio)
                loss  = self.model.get_loss_for_attack(x_adv)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self._project_delta()   # enforce L_inf constraint

                epoch_loss += loss.item()

            history['fooling_rates'].append(fooled_count / len(dataset))
            history['avg_losses'].append(epoch_loss / len(dataset))

        return self.delta.detach(), history

    
    def evaluate_single_sample(self, audio_tensor: torch.Tensor) -> dict:
        """Evaluate whether the current δ fools the model on a single sample."""
        with torch.no_grad():
            orig = self.model.transcribe(audio_tensor)
            adv  = self.model.transcribe(self._apply_perturbation(audio_tensor))
        return {'original': orig, 'adversarial': adv, 'is_fooled': orig != adv}


    