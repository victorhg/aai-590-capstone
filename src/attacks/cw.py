"""
Carlini-Wagner (CW) Attack Implementation for Whisper Audio ASR.

Formulation (targeted L2):
    minimize  ||delta||_2^2  +  c * CE(f(x + delta), target)

The optimizer (Adam + cosine LR) finds the smallest L2 perturbation that
causes Whisper to output the desired target phrase.

Binary search over ``c`` is performed to locate the smallest perturbation
that still achieves the attack — faithfully following the original
Carlini & Wagner (2017) algorithm.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class CWAuditoryAttack:
    """
    Targeted Carlini-Wagner attack for ASR models.

    Key improvements over the naïve version:
    - Binary search over ``c`` to find minimum-perturbation adversarial example.
    - Cosine annealing LR schedule for smoother convergence.
    - Random delta initialisation to escape flat loss regions.
    - Periodic transcription check with early stopping once target is found.
    """

    def __init__(
        self,
        whisper_model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.005,
        c: float = 50.0,
        steps: int = 500,
        binary_search_steps: int = 7,
        early_stop_check_every: int = 50,
        noise_init: float = 0.001,
        early_stop_budget: float = 0.6,
    ):
        """
        Args:
            whisper_model: A ``WhisperASRWithAttack`` instance exposing
                           ``get_loss_for_attack(audio, target_text)`` and
                           ``transcribe(audio)``.
            device: 'cpu', 'cuda', or 'mps'.
            learning_rate: Initial Adam step size.
            c: Starting regularisation constant (upper bound in binary search).
               Larger c → stronger push toward target phrase.
            steps: Inner optimisation iterations per binary-search round.
            binary_search_steps: Number of binary-search rounds over c.
            early_stop_check_every: Transcribe every N inner steps to check
                                    whether the target phrase is already achieved.
            noise_init: Magnitude of initial random perturbation.
            early_stop_budget: Fraction of steps after which we can early stop if successful.
        """
        self.model = whisper_model
        self.device = device
        self.learning_rate = learning_rate
        self.c = c
        self.steps = steps
        self.binary_search_steps = binary_search_steps
        self.early_stop_check_every = early_stop_check_every
        self.noise_init = noise_init
        self.early_stop_budget = early_stop_budget

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inner_optimization(
        self,
        audio: torch.Tensor,
        target_phrase: str,
        c_val: float,
    ):
        """
        One inner CW optimisation loop for a fixed ``c_val``.

        Returns:
            (best_adv, best_l2, found_success)
        """
        # Small random initialisation — avoids the flat-loss region at delta=0.
        delta = torch.randn_like(audio) * self.noise_init
        delta.requires_grad_(True)

        optimizer = optim.Adam([delta], lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.steps, eta_min=self.learning_rate * 0.01
        )

        best_adv = audio.clone().detach()
        best_l2 = float("inf")
        found_success = False

        for step in range(self.steps):
            optimizer.zero_grad()

            # Adversarial candidate — clamp to valid audio range.
            adv = torch.clamp(audio + delta, -1.0, 1.0)

            # CW loss: ||delta||^2 + c * CE(model(adv), target)
            l2_penalty = torch.sum(delta ** 2)
            attack_loss = self.model.get_loss_for_attack(adv, target_text=target_phrase)
            total_loss = l2_penalty + c_val * attack_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            l2_val = l2_penalty.item()

            # Periodic transcription check (cheaper than every step).
            is_check_step = step % self.early_stop_check_every == 0 or step == self.steps - 1
            if is_check_step:
                with torch.no_grad():
                    pred = self.model.transcribe(adv.squeeze(0))

                tqdm.write(
                    f"    step {step:4d}/{self.steps} | "
                    f"ce={attack_loss.item():.4f}  l2={l2_val:.5f}  "
                    f"pred='{pred[:70]}'"
                )

                if target_phrase.lower() in pred.lower():
                    found_success = True

            # Track best adversarial when successful
            if found_success and l2_val < best_l2:
                best_l2 = l2_val
                best_adv = adv.detach().clone()

            # Early stop if we found success and passed the budget
            if found_success and step > self.steps * self.early_stop_budget:
                tqdm.write("    ↳ early stop: target achieved with sufficient budget")
                break

        return best_adv, best_l2, found_success

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attack(self, audio_tensor: torch.Tensor, target_phrase: str) -> torch.Tensor:
        """
        Run CW optimisation with binary search over ``c``.

        Args:
            audio_tensor: Clean 1-D float32 waveform in ``[-1, 1]``.
            target_phrase: Text Whisper should output after the attack.

        Returns:
            Adversarial audio tensor (1-D, CPU, same length as input).
        """
        audio = audio_tensor.float().to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, T)

        # Binary search bounds for c.
        c_low = 0.0
        c_high = self.c * 2.0   # initial upper bound
        c_cur = self.c

        overall_best_adv = audio.clone().detach()
        overall_best_l2 = float("inf")

        for bs_step in range(self.binary_search_steps):
            tqdm.write(
                f"\n[CW] Binary-search round {bs_step + 1}/{self.binary_search_steps} "
                f"| c={c_cur:.4f}"
            )

            adv, l2, success = self._run_inner_optimization(audio, target_phrase, c_cur)

            if success:
                tqdm.write(f"  ✓ Succeeded  l2={l2:.5f}")
                if l2 < overall_best_l2:
                    overall_best_l2 = l2
                    overall_best_adv = adv
                # Attack worked → try a smaller c (smaller perturbation).
                c_high = c_cur
            else:
                tqdm.write("  ✗ Failed — increasing c")
                # Attack failed → increase c.
                c_low = c_cur

            c_cur = (c_low + c_high) / 2.0

        return overall_best_adv.squeeze(0).cpu()

    def batch_attack(
        self, audio_batch: torch.Tensor, target_phrase: str
    ) -> torch.Tensor:
        """
        Run CW on multiple utterances sequentially (memory-safe).

        Args:
            audio_batch: Tensor of shape (N, T).
            target_phrase: Shared target phrase.

        Returns:
            Adversarial batch tensor of shape (N, T).
        """
        results = [self.attack(audio, target_phrase) for audio in audio_batch]
        return torch.stack(results)


# Backward-compatible alias
CarliniWagnerAttack = CWAuditoryAttack

