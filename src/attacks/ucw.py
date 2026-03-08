"""
Universal Carlini-Wagner (UCW) Attack for ASR Models.
Fixed perturbation δ that forces a target phrase on any audio input.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import src.data as data_loader
from .base import BaseUniversalAttack
from .utils import prepare_audio


class UniversalCWAttack(BaseUniversalAttack):
    def __init__(
        self,
        whisper_model: nn.Module,
        target_phrase: str,
        uap_length: int,
        epsilon: float = 0.05,
        c: float = 1.0,
        learning_rate: float = 5e-4,
        device: str = "cpu",
        noise_init: float = 1e-4,
    ):
        self.model         = whisper_model
        self.target_phrase = target_phrase
        self.uap_length    = uap_length
        self.epsilon       = epsilon
        self.c             = c
        self.lr_init       = learning_rate
        self.device        = device

        # Initialize δ within L_inf epsilon-ball
        self.delta = (torch.randn(1, uap_length, device=device) * noise_init).clamp(-epsilon, epsilon)
        self.delta.requires_grad_(True)

    def _cw_loss(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute targeted CE loss for a single audio sample."""
        adv = self._apply_perturbation(audio)
        return self.c * self.model.get_loss_for_attack(adv, target_text=self.target_phrase)

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Cosine annealing schedule decaying to 1% of initial LR."""
        progress = step / max(total_steps, 1)
        return self.lr_init * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))

    def _step(self, global_step: int, total_steps: int):
        """Perform one PGD update step."""
        if self.delta.grad is None:
            return
        
        lr = self._get_lr(global_step, total_steps)
        with torch.no_grad():
            self.delta -= lr * self.delta.grad.sign()
            self._project_delta()
        
        self.delta.grad = None  # standard zero_grad

    def _validate(self, audio_files: list, limit: int = 20) -> float:
        """Calculate success rate on a subset of the training data."""
        successes = 0
        checked = 0
        
        with torch.no_grad():
            for fpath in audio_files[:limit]:
                try:
                    _, audio = data_loader.load_audio_tensor(fpath)
                    adv = self._apply_perturbation(prepare_audio(audio, self.device))
                    pred = self.model.transcribe(adv.squeeze(0))
                    if self.target_phrase.lower() in pred.lower():
                        successes += 1
                    checked += 1
                except Exception:
                    continue
        
        return successes / max(checked, 1)

    def train(self, audio_files: list, epochs: int = 10, batch_size: int = 1, grad_accum_steps: int = 1) -> dict:
        history = {"epoch_losses": [], "train_success_rates": []}
        
        # Calculate steps
        n_batches = max(1, len(audio_files) // batch_size)
        total_steps = epochs * (n_batches // grad_accum_steps)
        global_step = 0
        self.delta.grad = None

        for epoch in range(epochs):
            indices = np.random.permutation(len(audio_files))
            epoch_loss = 0.0
            accum_counter = 0

            with tqdm(range(0, len(audio_files), batch_size), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for start_idx in pbar:
                    # 1. Process Batch
                    batch_files = [audio_files[i] for i in indices[start_idx : start_idx + batch_size]]
                    batch_loss = torch.tensor(0.0, device=self.device)
                    
                    files_processed = 0
                    for fpath in batch_files:
                        try:
                            _, audio = data_loader.load_audio_tensor(fpath)
                            loss = self._cw_loss(prepare_audio(audio, self.device))
                            batch_loss = batch_loss + loss
                            files_processed += 1
                        except Exception:
                            continue

                    if files_processed == 0 or not batch_loss.requires_grad:
                        continue

                    # 2. Accumulate Gradients
                    (batch_loss / grad_accum_steps).backward()
                    epoch_loss += batch_loss.item()
                    accum_counter += 1

                    # 3. PGD Step (only if accumulation complete)
                    if accum_counter % grad_accum_steps == 0:
                        self._step(global_step, total_steps)
                        global_step += 1
                    
                    pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            # Flush any remaining gradients at end of epoch
            if accum_counter % grad_accum_steps != 0:
                self._step(global_step, total_steps)
                global_step += 1

            # 4. Validation & Logging
            sr = self._validate(audio_files)
            avg_loss = epoch_loss / max(len(audio_files), 1)
            
            history["epoch_losses"].append(avg_loss)
            history["train_success_rates"].append(sr)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                tqdm.write(f"  Result: loss={avg_loss:.4f} | success_rate={sr:.1%} | lr={self._get_lr(global_step, total_steps):.6f}")

        return history


