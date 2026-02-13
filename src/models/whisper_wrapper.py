import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor

# Whisper and Audio Constants
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_SAMPLES_30S = 480000  # 30 seconds at 16kHz
WHISPER_START_OF_TRANSCRIPT_TOKEN = 50258
AUDIO_MIN = -1.0
AUDIO_MAX = 1.0

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
            
        # Get correct mel filters from transformers to ensure accuracy
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        self.register_buffer("mel_filters", torch.from_numpy(feature_extractor.mel_filters).float().to(device))
        
        # Get processor for decoding tokens to text
        self.processor = WhisperProcessor.from_pretrained(model_path)
        
        # Standard Whisper constants
        self.n_fft = 400
        self.hop_length = 160
        self.n_samples = WHISPER_N_SAMPLES_30S
        self.expected_frames = 3000
    
    def _preprocess_audio(self, audio_tensor, requires_grad=True):
        """
        Preprocess audio tensor: ensure batch dimension, transfer to device,
        pad/crop to target length, and set requires_grad flag.
        
        Args:
            audio_tensor: Raw audio waveform (1D or 2D tensor)
            requires_grad: Whether to enable gradient computation
            
        Returns:
            Preprocessed audio tensor of shape (batch, n_samples)
        """
        # Ensure batch dimension
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Transfer to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Set gradient requirement
        if requires_grad and not audio_tensor.requires_grad:
            audio_tensor.requires_grad = True
        
        # Pad or crop to exactly n_samples (30 seconds)
        if audio_tensor.shape[1] < self.n_samples:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, self.n_samples - audio_tensor.shape[1])
            )
        else:
            audio_tensor = audio_tensor[:, :self.n_samples]
        
        return audio_tensor
    
    def _compute_mel_spectrogram(self, audio_tensor):
        """
        Compute differentiable log-mel spectrogram from preprocessed audio.
        This is the core transformation matching Whisper's preprocessing.
        
        Args:
            audio_tensor: Preprocessed audio tensor (batch, n_samples)
            
        Returns:
            Log-mel spectrogram tensor (batch, n_mels, n_frames)
        """
        # Compute STFT
        window = torch.hann_window(self.n_fft).to(self.device)
        stft = torch.stft(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True
        )
        
        # Power spectrum
        magnitudes = stft.abs() ** 2
        # Drop the last frame to get exactly 3000 frames (Whisper convention)
        magnitudes = magnitudes[:, :, :-1]
        
        # Project to Mel scale: (batch, freq, time) @ (freq, n_mels) -> (batch, time, n_mels)
        # magnitudes: (batch, 201, 3000) -> transpose to (batch, 3000, 201)
        # self.mel_filters: (201, 80)
        mels = torch.matmul(magnitudes.transpose(1, 2), self.mel_filters).transpose(1, 2)
        
        # Log-scaling and normalization (Whisper standard)
        log_mels = torch.log10(torch.clamp(mels, min=1e-10))
        log_mels = torch.maximum(log_mels, log_mels.max() - 8.0)
        log_mels = (log_mels + 4.0) / 4.0
        
        return log_mels
    
    @staticmethod
    def _clamp_audio(audio_tensor, min_val=AUDIO_MIN, max_val=AUDIO_MAX):
        """
        Clamp audio tensor to valid range.
        
        Args:
            audio_tensor: Audio tensor
            min_val: Minimum audio value (default: -1.0)
            max_val: Maximum audio value (default: 1.0)
            
        Returns:
            Clamped audio tensor
        """
        return torch.clamp(audio_tensor, min_val, max_val)

    def forward(self, audio_tensor):
        """
        Forward pass that accepts raw audio and returns logits.
        Ensures gradients flow back to audio_tensor.
        """
        # Preprocess audio: batch dimension, device, padding, requires_grad
        audio_tensor = self._preprocess_audio(audio_tensor, requires_grad=True)
        
        # Compute differentiable log-mel spectrogram
        log_mels = self._compute_mel_spectrogram(audio_tensor)
        
        # Forward pass through Whisper model
        # We need decoder_input_ids to get logits. We use <|startoftranscript|> token
        decoder_input_ids = torch.tensor(
            [[WHISPER_START_OF_TRANSCRIPT_TOKEN]] * audio_tensor.shape[0]
        ).to(self.device)
        
        # If we want to minimize 'CrossEntropy', we need logits over vocab.
        output = self.model(
            input_features=log_mels, 
            decoder_input_ids=decoder_input_ids
        )
            
        return output
    
    def transcribe(self, audio_tensor, max_length=448):
        """
        Generate transcription from audio tensor.
        Uses greedy decoding for efficiency.
        
        Args:
            audio_tensor: Raw audio waveform (1D or 2D tensor)
            max_length: Maximum number of tokens to generate
            
        Returns:
            str: Decoded transcription text
        """
        with torch.no_grad():
            # Preprocess audio (no gradients needed for transcription)
            audio_tensor = self._preprocess_audio(audio_tensor, requires_grad=False)
            
            # Compute mel spectrogram
            log_mels = self._compute_mel_spectrogram(audio_tensor)
            
            # Generate tokens using model's generate method
            predicted_ids = self.model.generate(
                log_mels,
                max_length=max_length,
                num_beams=1,  # Greedy decoding
                do_sample=False
            )
            
            # Decode to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
        return transcription.strip()
    
    def get_encoder_output(self, audio_tensor):
        """
        Get encoder output for computing attack loss.
        This is used when you need gradients through the encoder.
        
        Args:
            audio_tensor: Raw audio waveform with requires_grad=True
            
        Returns:
            encoder_output: Encoder hidden states (batch, 1500, hidden_dim)
        """
        # Preprocess audio with gradients enabled
        audio_tensor = self._preprocess_audio(audio_tensor, requires_grad=True)
        
        # Compute mel spectrogram
        log_mels = self._compute_mel_spectrogram(audio_tensor)
        
        # Get encoder output
        encoder_output = self.model.model.encoder(log_mels)
        
        return encoder_output.last_hidden_state
    
    def get_loss_for_attack(self, audio_tensor, target_text=None):
        """
        Compute loss for adversarial optimization.
        
        Args:
            audio_tensor: Raw audio with requires_grad=True
            target_text: Optional target transcription for targeted attacks
            
        Returns:
            loss: Scalar loss for backpropagation
        """
        if target_text is None:
            # Untargeted attack: Maximize encoder output uncertainty
            # This encourages the model to produce incoherent representations
            encoder_output = self.get_encoder_output(audio_tensor)
            
            # Compute entropy-like loss: minimize standard deviation across features
            # This makes the encoder output "flat" and less informative
            loss = -torch.mean(torch.std(encoder_output, dim=-1))
            
            return loss
        else:
            # Targeted attack: Minimize cross-entropy with target text
            # This is more complex and requires autoregressive generation
            raise NotImplementedError("Targeted attacks require autoregressive decoding - use untargeted for now")
