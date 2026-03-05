"""
Base class shared by all universal perturbation attacks (UAP, UCW).
"""
import torch
import torch.nn as nn

from .utils import tile_to_length, prepare_audio


class BaseUniversalAttack:
    """
    Common behaviour for universal perturbation attacks.

    Subclasses must set:
      - ``self.delta``   — ``(1, uap_length)`` perturbation tensor on ``self.device``
      - ``self.epsilon`` — L_inf bound
      - ``self.device``  — device string

    Inherited public API:
      ``apply(audio)`` → adversarial 1-D CPU tensor
      ``get_perturbation()`` → detached CPU clone of δ
      ``save(path)`` / ``load(path)`` — checkpoint δ
    """

    model: nn.Module
    delta: torch.Tensor
    epsilon: float
    device: str

    # ── Internals ──────────────────────────────────────────────────────────────

    def _effective_perturbation(self, audio: torch.Tensor) -> torch.Tensor:
        """Return δ cropped/tiled to match the current audio length."""
        return tile_to_length(self.delta, audio.shape[-1])

    def _apply_perturbation(self, audio: torch.Tensor) -> torch.Tensor:
        """Add δ to ``audio``, handling length mismatches, and clamp to [-1, 1]."""
        v = self._effective_perturbation(audio)
        return torch.clamp(audio + v, -1.0, 1.0)

    def _project_delta(self) -> None:
        """Project δ back onto the L_inf ε-ball (in-place, no gradient)."""
        with torch.no_grad():
            self.delta.clamp_(-self.epsilon, self.epsilon)

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add δ to ``audio`` and return the adversarial waveform on CPU.

        Args:
            audio: Clean 1-D or 2-D waveform tensor.

        Returns:
            1-D adversarial tensor on CPU.
        """
        with torch.no_grad():
            adv = self._apply_perturbation(prepare_audio(audio, self.device))
        return adv.squeeze(0).cpu()

    def get_perturbation(self) -> torch.Tensor:
        return self.delta.detach().clone().cpu()

    def save(self, path: str) -> None:
        torch.save(self.delta.detach().cpu(), path)
        print(f"Saved universal perturbation → {path}")

    def load(self, path: str) -> None:
        loaded = torch.load(path, map_location=self.device)
        with torch.no_grad():
            self.delta.copy_(loaded.to(self.device))
        print(f"Loaded universal perturbation ← {path}")
