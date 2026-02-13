"""
Model Registry for Transferability Testing

Provides a unified interface to load different ASR models:
- Whisper (OpenAI)
- Wav2Vec2 (Facebook)
- DeepSpeech (Mozilla) - optional if installed

Used for Week 4 transferability experiments.
"""

import torch
from typing import Tuple, Optional


class ModelRegistry:
    """
    Centralized registry for loading ASR models.
    """
    
    SUPPORTED_MODELS = {
        'whisper-tiny': 'openai/whisper-tiny',
        'whisper-base': 'openai/whisper-base',
        'whisper-small': 'openai/whisper-small',
        'whisper-medium': 'openai/whisper-medium',
        'wav2vec2-base': 'facebook/wav2vec2-base-960h',
        'wav2vec2-large': 'facebook/wav2vec2-large-960h',
        'wav2vec2-large-lv60': 'facebook/wav2vec2-large-960h-lv60-self',
    }
    
    @staticmethod
    def list_models():
        """List all supported models."""
        return list(ModelRegistry.SUPPORTED_MODELS.keys())
    
    @staticmethod
    def load_model(model_name: str, device: str = 'cpu'):
        """
        Load an ASR model by name.
        
        Args:
            model_name: Model identifier (e.g., 'whisper-base', 'wav2vec2-base')
            device: Device to load model on ('cpu', 'cuda', 'mps')
            
        Returns:
            model: Model wrapper with .transcribe() method
            
        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in ModelRegistry.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from: {ModelRegistry.list_models()}"
            )
        
        model_path = ModelRegistry.SUPPORTED_MODELS[model_name]
        
        if model_name.startswith('whisper'):
            return ModelRegistry._load_whisper(model_path, device)
        elif model_name.startswith('wav2vec2'):
            return ModelRegistry._load_wav2vec2(model_path, device)
        else:
            raise ValueError(f"Unknown model family: {model_name}")
    
    @staticmethod
    def _load_whisper(model_path: str, device: str):
        """Load Whisper model with attack wrapper."""
        from src.models.whisper_wrapper import WhisperASRWithAttack
        
        print(f"Loading Whisper model: {model_path}")
        model = WhisperASRWithAttack(model_path=model_path, device=device)
        print("✓ Whisper model loaded")
        
        return model
    
    @staticmethod
    def _load_wav2vec2(model_path: str, device: str):
        """Load Wav2Vec2 model with wrapper."""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        print(f"Loading Wav2Vec2 model: {model_path}")
        model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model.eval()
        print("✓ Wav2Vec2 model loaded")
        
        # Wrap in a simple interface
        return Wav2Vec2Wrapper(model, processor, device)


class Wav2Vec2Wrapper:
    """
    Wrapper for Wav2Vec2 models to provide a unified interface.
    """
    
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
    
    def transcribe(self, audio_tensor):
        """
        Transcribe audio using Wav2Vec2.
        
        Args:
            audio_tensor: Audio waveform (1D or 2D tensor)
            
        Returns:
            str: Transcription text
        """
        with torch.no_grad():
            # Ensure correct shape
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Move to CPU for processor (it expects numpy)
            audio_np = audio_tensor.cpu().numpy()
            
            # Process through model
            input_values = self.processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.to(self.device)
            
            # Get logits
            logits = self.model(input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
        return transcription.strip().lower()
    
    def get_loss_for_attack(self, audio_tensor, target_text=None):
        """
        Compute loss for adversarial attacks on Wav2Vec2.
        Note: This is a simplified version for untargeted attacks.
        """
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.squeeze(0)
        
        # Process through model
        audio_np = audio_tensor.detach().cpu().numpy()
        input_values = self.processor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(self.device)
        
        # Ensure gradients flow
        input_values.requires_grad = True
        
        # Get logits
        logits = self.model(input_values).logits
        
        if target_text is None:
            # Untargeted: maximize entropy (uncertainty)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            loss = -torch.mean(entropy)  # Maximize entropy = minimize negative entropy
        else:
            # Targeted: minimize cross-entropy with target
            raise NotImplementedError("Targeted attacks not implemented for Wav2Vec2")
        
        return loss
    
    def parameters(self):
        """Return model parameters (for compatibility)."""
        return self.model.parameters()


class TransferabilityTester:
    """
    Utility class for testing attack transferability across models.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
    
    def load_models(self, model_names):
        """
        Load multiple models for testing.
        
        Args:
            model_names: List of model identifiers
        """
        for name in model_names:
            print(f"\n--- Loading {name} ---")
            self.models[name] = ModelRegistry.load_model(name, self.device)
    
    def test_perturbation(self, audio_tensor, perturbation, audio_paths=None):
        """
        Test a perturbation across all loaded models.
        
        Args:
            audio_tensor: Clean audio tensor
            perturbation: Adversarial perturbation to apply
            audio_paths: Optional list of audio paths (for logging)
            
        Returns:
            results: Dictionary with per-model results
        """
        results = {}
        
        # Apply perturbation
        perturbed = torch.clamp(audio_tensor + perturbation, -1.0, 1.0)
        
        print("\n=== Transferability Test ===")
        
        for model_name, model in self.models.items():
            print(f"\nTesting on {model_name}...")
            
            # Get original transcription
            with torch.no_grad():
                orig_transcription = model.transcribe(audio_tensor)
            
            # Get adversarial transcription
            with torch.no_grad():
                adv_transcription = model.transcribe(perturbed)
            
            # Check if attack transferred
            is_fooled = (orig_transcription != adv_transcription)
            
            results[model_name] = {
                'original': orig_transcription,
                'adversarial': adv_transcription,
                'fooled': is_fooled
            }
            
            print(f"  Original:    '{orig_transcription}'")
            print(f"  Adversarial: '{adv_transcription}'")
            print(f"  Status: {'✓ FOOLED' if is_fooled else '✗ NOT FOOLED'}")
        
        # Calculate transfer rate
        fooled_count = sum(1 for r in results.values() if r['fooled'])
        transfer_rate = fooled_count / len(results) if results else 0
        
        print(f"\n--- Transfer Rate: {transfer_rate:.1%} ({fooled_count}/{len(results)}) ---")
        
        return results, transfer_rate
    
    def evaluate_dataset(self, dataset, perturbation, audio_length=480000):
        """
        Evaluate perturbation transferability on a full dataset.
        
        Args:
            dataset: List of audio file paths
            perturbation: Universal perturbation
            audio_length: Fixed audio length
            
        Returns:
            summary: Dictionary with aggregate results
        """
        from tqdm import tqdm
        from src.data.audio_loader import load_audio_tensor
        
        print(f"\n=== Evaluating Transferability on {len(dataset)} Samples ===")
        
        model_results = {name: {'fooled': 0, 'total': 0} for name in self.models.keys()}
        
        for audio_path in tqdm(dataset, desc="Processing samples"):
            try:
                # Load audio
                _, audio_tensor = load_audio_tensor(str(audio_path))
                audio_tensor = audio_tensor.to(self.device)
                
                # Pad/crop
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                if audio_tensor.shape[1] < audio_length:
                    audio_tensor = torch.nn.functional.pad(
                        audio_tensor, (0, audio_length - audio_tensor.shape[1])
                    )
                else:
                    audio_tensor = audio_tensor[:, :audio_length]
                
                # Test on each model
                perturbed = torch.clamp(audio_tensor + perturbation, -1.0, 1.0)
                
                for model_name, model in self.models.items():
                    with torch.no_grad():
                        orig = model.transcribe(audio_tensor)
                        adv = model.transcribe(perturbed)
                    
                    model_results[model_name]['total'] += 1
                    if orig != adv:
                        model_results[model_name]['fooled'] += 1
            
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        # Compute summary statistics
        summary = {}
        for model_name, counts in model_results.items():
            fooling_rate = counts['fooled'] / counts['total'] if counts['total'] > 0 else 0
            summary[model_name] = {
                'fooling_rate': fooling_rate,
                'fooled_count': counts['fooled'],
                'total_count': counts['total']
            }
            print(f"{model_name}: {fooling_rate:.1%} ({counts['fooled']}/{counts['total']})")
        
        return summary
