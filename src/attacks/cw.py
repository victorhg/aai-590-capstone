"""
Carlini-Wagner (CW) Attack for Audio (Whisper)
Targeted Attack implementation using L2 penalty.

The attack formulation:
minimize ||x_adv - x||_2^2 + c * f(x_adv)
where f(x_adv) = max(0, max_{y' != target} softmax(x_adv)[y'] - softmax(x_adv)[target])

This encourages the model to output the target class with high probability
while keeping the perturbation small in L2 norm.
"""

import torch
import torch.nn.functional as F
from src.data.audio_loader import AudioLoader

class CWAuditoryAttack:
    def __init__(self, whisper_model, device='cuda', learning_rate=0.01, c=1.0, steps=100, confidence=0.0):
        """
        Args:
            whisper_model: The loaded Whisper model.
            device: cuda or cpu.
            learning_rate: Optimization step size.
            c: Weighting of the f(x) term (must be positive).
            steps: Number of optimization iterations.
            confidence: Minimum confidence margin for target class.
        """
        self.model = whisper_model
        self.device = device
        self.learning_rate = learning_rate
        self.c = c
        self.steps = steps
        self.confidence = confidence

    def attack(self, clean_audio, target_tokens, max_epsilon=1.0, n_classes=51866):
        """
        Generates adversarial audio using CW attack.

        Args:
            clean_audio: Tensor of shape (1, T) or (T,).
            target_tokens: The target text to fool the model into producing.
            max_epsilon: L2 constraint radius (applied via sqrt).
            n_classes: Vocabulary size for model.

        Returns:
            adv_audio: The perturbed audio tensor.
        """
        # Ensure clean audio is on device and requires grad
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)
        
        clean_audio = clean_audio.to(self.device)
        clean_audio = clean_audio.requires_grad_(True)

        # Initialize perturbation with small random noise
        adv_audio = clean_audio + torch.randn_like(clean_audio) * (max_epsilon * 0.1)
        
        # Clip to valid audio range [-1, 1]
        adv_audio = torch.clamp(adv_audio, -1.0, 1.0)
        
        optimizer = torch.optim.Adam([adv_audio], lr=self.learning_rate)

        target_ids = self.model.tokenizer.encode(target_tokens).to(self.device)
        
        print(f"Starting CW attack (Target: '{target_tokens}')...")

        for step in range(self.steps):
            optimizer.zero_grad()
            
            # Forward pass
            # Whisper expects (1, seq_len). We add batch dim.
            mel = self.model.log_mel_spectrogram(adv_audio.unsqueeze(0))
            
            # Encode audio to get logits
            with torch.no_grad():
                # Get logits for the input features
                encoder_out = self.model.encoder(mel)
                # For a robust targeted attack, we usually need to run the decoder,
                # but calculating decoder loss requires target tokens length matching.
                # To simplify for this implementation, we focus on the Encoder-Decoder 
                # log-probability if possible, or just Encoder if we want a simpler 
                # "uncertainty" attack.
                # However, CW usually targets the *output*.
                # We will try to maximize the logprob of the target sequence using the Decoder.
                logits, _, _ = self.model.decoder(encoder_out, mel)
            
            # Cross Entropy Loss (Targeted: Minimize Loss(y_target))
            # We create a mask to only consider the target tokens
            # Note: This is a simplified approach. In practice, handling variable length 
            # target sequences in a CW loop is complex.
            
            loss = 0
            for token in target_ids:
                loss += F.cross_entropy(logits, token.unsqueeze(0))

            loss = -loss # We want to Maximize the probability of target tokens
            
            # CW Loss formulation:
            # L2 ||x_adv - x||_2^2 + c * f(x_adv)
            # where f(x_adv) is the margin (usually negative for targeted)
            
            l2_dist = torch.sum((adv_audio - clean_audio) ** 2)
            
            total_loss = loss + (self.c * l2_dist)

            total_loss.backward()
            optimizer.step()

            # Update and Clip
            adv_audio = torch.clamp(adv_audio, -1.0, 1.0)
            
            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{self.steps}, Loss: {total_loss.item():.4f}")

        print("CW Attack finished.")
        return adv_audio.detach()

