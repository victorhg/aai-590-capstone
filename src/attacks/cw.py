"""
Carlini-Wagner (CW) Attack Implementation for Whisper Audio ASR.

Formulation (targeted L2):
    minimize  ||delta||_2^2  +  c * CE(f(x + delta), target)

The optimizer (Adam) naturally finds the smallest L2 perturbation that
causes Whisper to output the desired target phrase.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class CWAuditoryAttack:
    """
    Targeted Carlini-Wagner attack for ASR models.

    Constructor matches the notebook interface:
        CWAuditoryAttack(whisper_model, device, learning_rate, c, steps)
    """

    def __init__(
        self,
        whisper_model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.01,
        c: float = 1.0,
        steps: int = 100,
    ):
        """
        Args:
            whisper_model: A ``WhisperASRWithAttack`` instance (or any model that
                           exposes ``get_loss_for_attack(audio, target_text)`` and
                           ``transcribe(audio)``).
            device: 'cpu', 'cuda', or 'mps'.
            learning_rate: Adam step size for the perturbation.
            c: Regularization constant balancing L2 size vs. attack strength.
               Larger c → stronger push toward the target; smaller c → smaller
               perturbation but attack may not converge.
            steps: Number of Adam optimization iterations.
        """
        self.model = whisper_model
        self.device = device
        self.learning_rate = learning_rate
        self.c = c
        self.steps = steps

        # Ensure model stays in eval mode with frozen weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attack(self, audio_tensor: torch.Tensor, target_phrase: str) -> torch.Tensor:
        """
        Run CW optimization to produce adversarial audio.

        Args:
            audio_tensor: Clean 1-D float32 waveform in ``[-1, 1]``.
            target_phrase: Text string Whisper should output after the attack.

        Returns:
            Adversarial audio tensor (1-D, CPU, same length as input).
        """
        # Move original audio to device; keep a CPU copy for the return value.
        audio = audio_tensor.float().to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, T)

        # Initialise perturbation δ at zero (imperceptible starting point).
        delta = torch.zeros_like(audio, requires_grad=True)
        optimizer = optim.Adam([delta], lr=self.learning_rate)

        best_adv = audio.clone().detach()
        best_loss = float("inf")

        for step in tqdm(range(self.steps), desc="CW Attack", leave=False):
            optimizer.zero_grad()

            # Adversarial candidate — clamped to valid audio range.
            adv = torch.clamp(audio + delta, -1.0, 1.0)

            # CW loss: L2(δ)² + c · CE(f(adv), target)
            l2_penalty = torch.sum(delta ** 2)
            attack_loss = self.model.get_loss_for_attack(adv, target_text=target_phrase)
            total_loss = l2_penalty + self.c * attack_loss

            total_loss.backward()
            optimizer.step()

            # Track best (lowest total) loss for optional early stopping.
            loss_val = total_loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_adv = adv.detach().clone()

            if step % max(1, self.steps // 10) == 0:
                tqdm.write(f"  step {step:4d}/{self.steps} | loss={loss_val:.4f} "
                           f"(l2={l2_penalty.item():.4f}, ce={attack_loss.item():.4f})")

        return best_adv.squeeze(0).cpu()

    def batch_attack(
        self, audio_batch: torch.Tensor, target_phrase: str
    ) -> torch.Tensor:
        """
        Run CW on multiple utterances sequentially.

        Args:
            audio_batch: Tensor of shape (N, T).
            target_phrase: Shared target phrase for all utterances.

        Returns:
            Adversarial batch tensor of shape (N, T).
        """
        results = [self.attack(audio, target_phrase) for audio in audio_batch]
        return torch.stack(results)


# Backward-compatible alias
CarliniWagnerAttack = CWAuditoryAttack
