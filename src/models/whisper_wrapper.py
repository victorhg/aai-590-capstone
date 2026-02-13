import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor

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
        self.n_samples = 480000  # 30 seconds at 16kHz
        self.expected_frames = 3000

    def forward(self, audio_tensor):
        """
        Forward pass that accepts raw audio and returns logits.
        Ensures gradients flow back to audio_tensor.
        """
        # audio_tensor shape: (batch, time) or (time)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        # Ensure requires_grad is True so we can compute dLoss/dAudio
        if not audio_tensor.requires_grad:
            audio_tensor.requires_grad = True
        
        # 1. Pad/Crop audio to exactly 30 seconds (480,000 samples)
        # This is CRITICAL for Whisper's fixed-length encoder
        if audio_tensor.shape[1] < self.n_samples:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.n_samples - audio_tensor.shape[1]))
        else:
            audio_tensor = audio_tensor[:, :self.n_samples]

        # 2. Differentiable Log-Mel Spectrogram (Matching Whisper Spec)
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
        # Drop the last frame to get exactly 3000 frames (Whisper convention for 480k samples)
        magnitudes = magnitudes[:, :, :-1]
        
        # Project to Mel scale
        # safe matmul on MPS: (batch, time, freq) @ (freq, n_mels) -> (batch, time, n_mels)
        # magnitudes: (batch, 201, 3000) -> transpose to (batch, 3000, 201)
        # self.mel_filters: (201, 80)
        mels = torch.matmul(magnitudes.transpose(1, 2), self.mel_filters).transpose(1, 2)
        
        # Log-scaling and Normalization
        log_mels = torch.log10(torch.clamp(mels, min=1e-10))
        # Whisper normalization: scale based on max value in the segment
        log_mels = torch.maximum(log_mels, log_mels.max() - 8.0)
        log_mels = (log_mels + 4.0) / 4.0
        
        # 3. Forward Pass
        # We need decoder_input_ids to get logits. We use <|startoftranscript|> (50258)
        decoder_input_ids = torch.tensor([[50258]] * audio_tensor.shape[0]).to(self.device)
        
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
            # Get mel features using forward pass
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            # Compute input features
            if audio_tensor.shape[1] < self.n_samples:
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.n_samples - audio_tensor.shape[1]))
            else:
                audio_tensor = audio_tensor[:, :self.n_samples]
            
            # Compute mel spectrogram
            window = torch.hann_window(self.n_fft).to(self.device)
            stft = torch.stft(
                audio_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                center=True,
                return_complex=True
            )
            magnitudes = stft.abs() ** 2
            magnitudes = magnitudes[:, :, :-1]
            mels = torch.matmul(magnitudes.transpose(1, 2), self.mel_filters).transpose(1, 2)
            log_mels = torch.log10(torch.clamp(mels, min=1e-10))
            log_mels = torch.maximum(log_mels, log_mels.max() - 8.0)
            log_mels = (log_mels + 4.0) / 4.0
            
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
        # Process through forward pass
        output = self.forward(audio_tensor)
        
        # Extract encoder output from the model output
        # For WhisperForConditionalGeneration, we need to run encoder separately
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        if not audio_tensor.requires_grad:
            audio_tensor.requires_grad = True
        
        # Pad/Crop
        if audio_tensor.shape[1] < self.n_samples:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.n_samples - audio_tensor.shape[1]))
        else:
            audio_tensor = audio_tensor[:, :self.n_samples]
        
        # Compute mel spectrogram
        window = torch.hann_window(self.n_fft).to(self.device)
        stft = torch.stft(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True
        )
        magnitudes = stft.abs() ** 2
        magnitudes = magnitudes[:, :, :-1]
        mels = torch.matmul(magnitudes.transpose(1, 2), self.mel_filters).transpose(1, 2)
        log_mels = torch.log10(torch.clamp(mels, min=1e-10))
        log_mels = torch.maximum(log_mels, log_mels.max() - 8.0)
        log_mels = (log_mels + 4.0) / 4.0
        
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
