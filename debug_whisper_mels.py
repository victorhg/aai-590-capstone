
import torch
import numpy as np
from transformers import WhisperFeatureExtractor

def test_whisper_dimensions():
    print("--- Testing Whisper Feature Extractor Dimensions ---")
    model_path = "openai/whisper-base"
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading feature extractor: {e}")
        return

    mel_filters = feature_extractor.mel_filters
    print(f"feature_extractor.mel_filters shape: {mel_filters.shape}")
    
    # Expected: (80, 201) usually (n_mels, n_freq)
    
    # Simulate Audio
    sr = 16000
    seconds = 30
    audio = np.random.randn(sr * seconds).astype(np.float32)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    
    print(f"Audio Tensor Shape: {audio_tensor.shape}")
    
    # Manual STFT replication attempt (from wrapper code)
    n_fft = 400
    hop_length = 160
    window = torch.hann_window(n_fft)
    
    # 1. Pad/Crop to 30s
    if audio_tensor.shape[1] < 480000:
         audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 480000 - audio_tensor.shape[1]))
    else:
         audio_tensor = audio_tensor[:, :480000]

    stft = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True
    )
    magnitudes = stft.abs() ** 2
    # magnitudes shape: (Batch, Freq, Time) = (1, 201, 3001) usually for center=True 480000 samples
    
    print(f"STFT Magnitudes Shape (center=True): {magnitudes.shape}")
    
    # Wrapper code does magnitudes[:, :, :-1]
    magnitudes = magnitudes[:, :, :-1]
    print(f"Magnitudes after slicing :-1: {magnitudes.shape}") # Should be (1, 201, 3000)

    # Convert mel_filters to tensor
    mel_filters_tensor = torch.from_numpy(mel_filters).float()
    
    # Wrapper Logic:
    # mels = torch.matmul(magnitudes.transpose(1, 2), self.mel_filters).transpose(1, 2)
    # magnitudes.transpose(1, 2) -> (1, 3000, 201)
    
    print(f"Look at matmul: (1, 3000, 201) @ {mel_filters_tensor.shape}")
    
    try:
        mels = torch.matmul(magnitudes.transpose(1, 2), mel_filters_tensor).transpose(1, 2)
        print("Matmul Successful!")
        print(f"Mels Shape: {mels.shape}")
    except RuntimeError as e:
        print(f"Matmul Failed: {e}")
        print("Trying with Transpose of filters...")
        try:
             mels = torch.matmul(magnitudes.transpose(1, 2), mel_filters_tensor.T).transpose(1, 2)
             print("Matmul with .T Successful!")
             print(f"Mels Shape: {mels.shape}")
        except RuntimeError as e2:
             print(f"Matmul with .T Failed: {e2}")

    # Log Logic Check
    # The feature extractor output
    print("\n--- Comparing with HF execute ---")
    hf_out = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    hf_mels = hf_out.input_features
    print(f"HF Output Shape: {hf_mels.shape}")
    
    if 'mels' in locals():
        # Complete the manual process to compare values
        log_mels = torch.log10(torch.clamp(mels, min=1e-10))
        log_mels = torch.maximum(log_mels, log_mels.max() - 8.0)
        log_mels = (log_mels + 4.0) / 4.0
        
        print("\n--- Value Comparison ---")
        print(f"Manual Mean: {log_mels.mean().item():.4f}, Max: {log_mels.max().item():.4f}, Min: {log_mels.min().item():.4f}")
        print(f"HF Mean: {hf_mels.mean().item():.4f}, Max: {hf_mels.max().item():.4f}, Min: {hf_mels.min().item():.4f}")
        
        # Check if identical (unlikely to be exactly identical due to float/implementation diffs, but should be close)
        # Note: HF implementation padding logic is slightly different (reflect vs constant, center=False)
        # HF does:
        # waveform = np.pad(waveform, ...)
        # window = np.hanning(n_fft)
        # stft = np.librosa.stft(..., center=True, pad_mode="reflect") <--- WAIT, HF uses center=True?
        
        # Actually HF implementation details:
        # self.feature_extractor(raw_speech) calls `_compute_log_mel_spectrogram`
        # which calls `stft(..., center=True)` ?
        
        # feature_extractor class says:
        # padding_side = "right"
        # padding_value = 0.0
        
        pass

if __name__ == "__main__":
    test_whisper_dimensions()
