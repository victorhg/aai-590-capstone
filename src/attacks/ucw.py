"""
Universal Carlini-Wagner (UCW) Attack for ASR Models.

Finds a single, fixed perturbation δ that, when added to any audio clip,
forces the target ASR model to output a specific ``target_phrase``.

Optimization objective (targeted, summed over a batch — Zhang et al. 2021):

    L(δ) = Σ_i  CE( f(x_i + δ'), y_target )

δ is updated via **PGD sign-gradient descent** and projected back onto the
L_inf ε-ball after each step (AGENTS.md §5: explicit clamp to avoid
clipping artifacts).  The L_inf projection replaces the original L2 penalty
which was counterproductive under a tight ε budget.

References:
    • Neekhara et al. 2019 — "Universal Adversarial Perturbations for Speech
      Recognition Systems", Interspeech 2019. https://arxiv.org/abs/1905.03828
    • Zhang et al. 2021    — Batch CW + PGD universal attack.
      https://ar5iv.labs.arxiv.org/html/2105.09022
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
    - **PGD sign-gradient** updates instead of Adam for L_inf geometry.
    - **No L2 penalty** — L_inf projection alone bounds the perturbation.
    - **Cosine-annealing** learning-rate schedule for stable convergence.

    Inherits ``apply``, ``get_perturbation``, ``save``, and ``load`` from
    :class:`~attacks.base.BaseUniversalAttack`.

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
        epsilon: float = 0.05,
        c: float = 1.0,
        learning_rate: float = 5e-4,
        device: str = "cpu",
        noise_init: float = 1e-4,
    ):
        """
        Args:
            whisper_model : ``WhisperASRWithAttack`` instance.
            target_phrase : Text that the model should produce on every input.
            uap_length    : Number of samples in the universal perturbation.
                            Recommended: 10–30 s × 16 000 Hz.
            epsilon       : L_inf constraint — keeps ``||δ||_inf ≤ epsilon``.
                            Higher ε → higher success rate but more audible.
                            Use ≥ 0.05 for targeted universal attacks.
            c             : Scalar multiplier on the CE loss (usually 1.0
                            since the L2 penalty has been removed).
            learning_rate : Step size for PGD sign-gradient updates.
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
        self.lr_init       = learning_rate
        self.device        = device

        # Initialise δ inside the ε-ball with small random noise.
        self.delta = torch.randn(1, uap_length, device=device) * noise_init
        self.delta = self.delta.clamp(-epsilon, epsilon)
        self.delta.requires_grad_(True)

    def _cw_loss_single(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Loss for one audio clip (CE only — L_inf projection handles constraints):

            c · CE( f(x + δ'), target )

        The L2 penalty is intentionally removed: under a tight L_inf budget the
        L2 term was consuming ~50 % of the total loss and pushing δ → 0,
        actively fighting the adversarial objective.
        """
        adv = self._apply_perturbation(audio)
        attack_loss = self.model.get_loss_for_attack(adv, target_text=self.target_phrase)
        return self.c * attack_loss

    # ── LR schedule helpers ─────────────────────────────────────────────────────

    def _cosine_lr(self, step: int, total_steps: int) -> float:
        """Cosine-annealing learning rate (decays to 1 % of initial LR)."""
        return self.lr_init * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1))))

    # ── training ───────────────────────────────────────────────────────────────

    def train(
        self,
        audio_files: list,
        epochs: int = 10,
        batch_size: int = 1,
        grad_accum_steps: int = 1,
    ) -> dict:
        """
        Train δ over *epochs* passes of *audio_files* using PGD
        sign-gradient descent with cosine-annealing LR.

        Args:
            audio_files      : List of .flac / .wav paths (training set).
            epochs           : Number of full passes over the training set.
            batch_size       : Clips per gradient step.  Keep ≤ 4 on consumer
                               GPUs to avoid OOM (AGENTS.md §3).
            grad_accum_steps : Number of mini-batches to accumulate gradients
                               over before performing a single PGD step.
                               Effective batch = batch_size × grad_accum_steps.

        Returns:
            ``history`` dict with keys:
              - ``epoch_losses``        — average loss per epoch
              - ``train_success_rates`` — fraction of first-20 train samples
                                          where the target phrase appears
        """
        history = {"epoch_losses": [], "train_success_rates": []}

        n_batches_per_epoch = max(1, len(audio_files) // batch_size)
        total_steps = epochs * (n_batches_per_epoch // grad_accum_steps)
        global_step = 0

        for epoch in range(epochs):
            epoch_loss    = 0.0
            success_count = 0
            n_evaluated   = 0
            accum_count   = 0

            # Shuffle for diversity each epoch.
            indices = np.random.permutation(len(audio_files))

            # Zero accumulated gradients at start of epoch
            if self.delta.grad is not None:
                self.delta.grad.zero_()

            with tqdm(
                range(0, len(audio_files), batch_size),
                leave=True,
            ) as pbar:
                for batch_start in pbar:
                    batch_idx   = indices[batch_start : batch_start + batch_size]
                    batch_files = [audio_files[i] for i in batch_idx]

                    batch_loss = torch.tensor(0.0, device=self.device)

                    for fpath in batch_files:
                        try:
                            _, audio = data_loader.load_audio_tensor(fpath)
                        except Exception:
                            continue

                        batch_loss = batch_loss + self._cw_loss_single(
                            prepare_audio(audio, self.device)
                        )

                    # Scale loss for gradient accumulation
                    (batch_loss / grad_accum_steps).backward()
                    accum_count += 1

                    epoch_loss += batch_loss.item()
                    pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

                    # ── PGD sign-gradient step after accumulating ──────────
                    if accum_count % grad_accum_steps == 0:
                        lr_t = self._cosine_lr(global_step, total_steps)
                        with torch.no_grad():
                            self.delta -= lr_t * self.delta.grad.sign()
                            self._project_delta()   # enforce L_inf constraint
                        self.delta.grad.zero_()
                        global_step += 1

            # ── flush any remaining accumulated gradients ──────────────────
            if accum_count % grad_accum_steps != 0:
                lr_t = self._cosine_lr(global_step, total_steps)
                with torch.no_grad():
                    self.delta -= lr_t * self.delta.grad.sign()
                    self._project_delta()
                self.delta.grad.zero_()
                global_step += 1

            # ── end-of-epoch success-rate proxy on first 20 train samples ──
            with torch.no_grad():
                for fpath in audio_files[:20]:
                    try:
                        _, audio = data_loader.load_audio_tensor(fpath)
                    except Exception:
                        continue

                    adv  = self._apply_perturbation(prepare_audio(audio, self.device))
                    pred = self.model.transcribe(adv.squeeze(0))
                    if self.target_phrase.lower() in pred.lower():
                        success_count += 1
                    n_evaluated += 1

            sr   = success_count / max(n_evaluated, 1)
            avgl = epoch_loss / max(len(audio_files), 1)
            history["epoch_losses"].append(avgl)
            history["train_success_rates"].append(sr)
            if epoch % 50 == 0 or epoch == epochs - 1:
                tqdm.write(
                    f"  Epoch {epoch + 1:2d}/{epochs} | "
                    f"avg_loss={avgl:.4f} | train_success_rate={sr:.1%} | "
                    f"lr={self._cosine_lr(global_step, total_steps):.6f}"
                )

        return history


