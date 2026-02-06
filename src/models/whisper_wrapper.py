import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperForConditionalGeneration

class WhisperASRWithAttack(nn.Module):
    """
    Wrapper for Whisper to compute gradients w.r.t input audio.
    Handling the Differentiable Mel Spectrogram calculation manually.
    """
    def __init__(self, model_path="openai/whisper-base", device="cpu"):
        super().__init__()
        self.device = device
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        self.model.eval()
        
        # Freeze model weights
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Differentiable Mel Spectrogram Layer using Torchaudio
        # Matches Whisper specs: 16kHz, 80 mel filters, N_FFT 400, HOP 160
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
        ).to(device)

    def forward(self, audio_tensor):
        """
        Forward pass that accepts raw audio and returns logits.
        Ensures gradients flow back to audio_tensor.
        """
        # audio_tensor shape: (batch, time) or (time)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Ensure requires_grad is True so we can compute dLoss/dAudio
        if not audio_tensor.requires_grad:
            audio_tensor.requires_grad = True
        
        # 1. Compute Log-Mel Spectrogram (Differentiable)
        mels = self.mel_transform(audio_tensor)
        log_mels = torch.log10(torch.clamp(mels, min=1e-10))
        
        # 2. Adjust for Whisper Input
        # Note: In a full pipeline, we should Pad/Crop to 30s here.
        
        # 3. Forward Pass
        # We need decoder_input_ids to get logits. We use <|startoftranscript|> (50258)
        decoder_input_ids = torch.tensor([[50258]] * audio_tensor.shape[0]).to(self.device)
        
        # HF WhisperModel returns 'last_hidden_state' usually if we don't ask for logits?
        # No, WhisperModel is the encoder-decoder. It returns Seq2SeqModelOutput.
        # But wait, WhisperModel usually outputs 'last_hidden_state' of decoder?
        # If we want logits (probabilities over vocabulary), we might need WhisperForConditionalGeneration!
        # Good catch. WhisperModel output might be raw vectors, not token probs.
        # Checking... WhisperForConditionalGeneration is the one with the LM Head.
        # WhisperModel is just the transformer.
        
        # If we want to minimize 'CrossEntropy', we need logits over vocab.
        # I should assume we strictly need WhisperForConditionalGeneration.
        
        output = self.model(
            input_features=log_mels, 
            decoder_input_ids=decoder_input_ids
        )
            
        return output
