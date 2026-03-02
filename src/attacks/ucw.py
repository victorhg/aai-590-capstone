"""
Universal Carlini-Wagner (UCW) Attack for ASR Models.

Finds a single, fixed perturbation δ that, when added to any audio clip,
forces the target ASR model to output a specific ``target_phrase``.

Optimization objective (targeted, summed over a batch — Zhang et al. 2021):

    L(δ) = Σ_i  [ ||δ||_2^2  +  c · CE( f(x_i + δ'), y_target ) ]

δ is updated with Adam and projected back onto the L_inf ε-ball after each
gradient step (AGENTS.md §5: explicit clamp to avoid clipping artifacts).

References:
    • Neekhara et al. 2019 — "Universal Adversarial Perturbations for Speech
      Recognition Systems", Interspeech 2019. https://arxiv.org/abs/1905.03828
    • Zhang et al. 2021    — Batch CW + PGD universal attack.
      https://ar5iv.labs.arxiv.org/html/2105.09022
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import src.data as data_loader


class UniversalCWAttack:
    """
    Universal Carlini-Wagner attack for ASR models.

    A **single, fixed** perturbation δ is trained over a diverse set of audio
    clips so that adding δ to any clip causes the ASR model to output
    ``target_phrase``.

    Key design decisions (AGENTS.md):
    - ``requires_grad=True`` is set on the audio tensor **before** the
      differentiable Mel-spectrogram computation → full gradient flow.
    - All optimisation is performed in the **16 kHz** domain; no resampling.
    - Default ``batch_size=1`` to avoid OOM on consumer GPUs.
    - ``torch.clamp(audio + δ, -1, 1)`` in every forward pass prevents clipping.
    - Length-mismatch handler crops δ when audio < δ and tiles when audio > δ.

    Usage::

        ucw = UniversalCWAttack(model, target_phrase="hello world", uap_length=80000)
        history = ucw.train(audio_files, epochs=10)
        adv = ucw.apply(clean_audio_tensor)
        ucw.save("results/ucw_delta.pt")
    """

    def __init__(
        self,
        whisper_model: nn.Module,
        target_phrase: str,
        uap_length: int,
        epsilon: float = 0.02,
        c: float = 50.0,
        learning_rate: float = 0.005,
        device: str = "cpu",
        noise_init: float = 1e-4,
    ):
        """
        Args:
            whisper_model : ``WhisperASRWithAttack`` instance.
            target_phrase : Text that the model should produce on every input.
            uap_length    : Number of samples in the universal perturbation.
                            Recommended: 5 s × 16 000 Hz = 80 000 samples.
            epsilon       : L_inf constraint — keeps ``||δ||_inf ≤ epsilon``.
                            Higher ε → higher success rate but more audible.
            c             : Weight of the cross-entropy term in the CW loss.
                            Larger c → stronger push toward target_phrase.
            learning_rate : Adam learning rate.
            device        : ``'cpu'``, ``'cuda'``, or ``'mps'``.
            noise_init    : Std-dev of random initialisation for δ (avoids
                            the flat-loss region at δ = 0).
        """
        self.model         = whisper_model
        self.target_phrase = target_phrase
        self.uap_length    = uap_length
        self.epsilon       = epsilon
        self.c             = c
        self.lr            = learning_rate
        self.device        = device

        # Initialise δ inside the ε-ball with small random noise.
        self.delta = torch.randn(1, uap_length, device=device) * noise_init
        self.delta = self.delta.clamp(-epsilon, epsilon)
        self.delta.requires_grad_(True)

        self.optimizer = optim.Adam([self.delta], lr=self.lr)

    def _apply_perturbation(self, audio: torch.Tensor) -> torch.Tensor:
        audio_len = audio.shape[-1]
        uap_len   = self.delta.shape[-1]

        if audio_len < uap_len:
            v = self.delta[:, :audio_len]
        elif audio_len > uap_len:
            repeats = (audio_len + uap_len - 1) // uap_len
            v = self.delta.repeat(1, repeats)[:, :audio_len]
        else:
            v = self.delta

        # Explicit clamp — prevents clipping artifacts 
        return torch.clamp(audio + v, -1.0, 1.0)

    def _project_delta(self):
        """Project δ back onto the L_inf ε-ball (in-place, no gradient)."""
        with torch.no_grad():
            self.delta.clamp_(-self.epsilon, self.epsilon)

    def _cw_loss_single(self, audio: torch.Tensor) -> torch.Tensor:
        """
        CW loss for one audio clip:

            ||δ||_2^2  +  c · CE( f(x + δ'), target )
        """
        adv = self._apply_perturbation(audio)
        l2_penalty  = torch.sum(self.delta ** 2)
        attack_loss = self.model.get_loss_for_attack(adv, target_text=self.target_phrase)
        return l2_penalty + self.c * attack_loss

    # ── training ───────────────────────────────────────────────────────────────

    def train(
        self,
        audio_files: list,
        epochs: int = 10,
        batch_size: int = 1,
    ) -> dict:
        """
        Train δ over *epochs* passes of *audio_files*.

        Args:
            audio_files : List of .flac / .wav paths (training set).
            epochs      : Number of full passes over the training set.
            batch_size  : Clips per gradient step.  Keep ≤ 4 on consumer GPUs
                          to avoid OOM (AGENTS.md §3).

        Returns:
            ``history`` dict with keys:
              - ``epoch_losses``       — average CW loss per epoch
              - ``train_success_rates`` — fraction of first-20 train samples
                                         where the target phrase appears
        """
        history = {"epoch_losses": [], "train_success_rates": []}

        for epoch in range(epochs):
            epoch_loss    = 0.0
            success_count = 0
            n_evaluated   = 0

            # Shuffle for diversity each epoch.
            indices = np.random.permutation(len(audio_files))

            with tqdm(
                range(0, len(audio_files), batch_size),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
            ) as pbar:
                for batch_start in pbar:
                    batch_idx   = indices[batch_start : batch_start + batch_size]
                    batch_files = [audio_files[i] for i in batch_idx]

                    self.optimizer.zero_grad()
                    batch_loss = torch.tensor(0.0, device=self.device)

                    for fpath in batch_files:
                        try:
                            _, audio = data_loader.load_audio_tensor(fpath)
                        except Exception:
                            continue

                        if audio.ndim == 1:
                            audio = audio.unsqueeze(0)
                        audio = audio.to(self.device)

                        batch_loss = batch_loss + self._cw_loss_single(audio)

                    batch_loss.backward()
                    self.optimizer.step()
                    self._project_delta()   # enforce L_inf constraint

                    epoch_loss += batch_loss.item()
                    pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            # ── end-of-epoch success-rate proxy on first 20 train samples ──────
            with torch.no_grad():
                for fpath in audio_files[:20]:
                    try:
                        _, audio = data_loader.load_audio_tensor(fpath)
                    except Exception:
                        continue

                    if audio.ndim == 1:
                        audio = audio.unsqueeze(0)
                    audio = audio.to(self.device)

                    adv  = self._apply_perturbation(audio)
                    pred = self.model.transcribe(adv.squeeze(0))
                    if self.target_phrase.lower() in pred.lower():
                        success_count += 1
                    n_evaluated += 1

            sr   = success_count / max(n_evaluated, 1)
            avgl = epoch_loss / max(len(audio_files), 1)
            history["epoch_losses"].append(avgl)
            history["train_success_rates"].append(sr)

            tqdm.write(
                f"  Epoch {epoch + 1:2d}/{epochs} | "
                f"avg_loss={avgl:.4f} | train_success_rate={sr:.1%}"
            )

        return history

    def get_perturbation(self) -> torch.Tensor:
        return self.delta.detach().clone().cpu()

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add δ to a single audio tensor and return adversarial audio (CPU).
        Returns:
            1-D adversarial tensor on CPU.
        """
        with torch.no_grad():
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            adv = self._apply_perturbation(audio.to(self.device))
        return adv.squeeze(0).cpu()

    def save(self, path: str):
        torch.save(self.delta.detach().cpu(), path)
        print(f"Saved universal perturbation → {path}")

    def load(self, path: str):
        loaded = torch.load(path, map_location=self.device)
        with torch.no_grad():
            self.delta.copy_(loaded.to(self.device))
        print(f"Loaded universal perturbation ← {path}")
