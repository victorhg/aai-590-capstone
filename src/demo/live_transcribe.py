import gradio as gr
import torch
import numpy as np
import soundfile as sf
import os
from pathlib import Path

# Attempt to import the project modules
try:
    from src.models.whisper_wrapper import load_model, transcribe_audio
    from src.attacks.uap import UniversalPerturbation
except ImportError:
    print("Warning: Project modules not found. Ensure running in correct environment.")
    pass

# --- Configuration ---
MODEL_SIZE = "base"  # or "small", "medium", "large"
FPS = 30
DEFAULT_EPSILON = 0.02
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds for Whisper

class LiveTranscriber:
    def __init__(self, model_size=MODEL_SIZE):
        self.model = load_model(model_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load UAP if available
        self.uap_path = Path("results/universal_perturbation_v.pt")
        self.uap = UniversalPerturbation.load(self.uap_path) if self.uap_path.exists() else None
        
        print(f"Live Transcriber initialized on {self.device}")

    def apply_attack(self, audio_chunk, mode="clean"):
        """
        Apply perturbation based on mode.
        mode: 'clean', 'uap', 'targeted'
        """
        # Ensure float32 and -1.0 to 1.0 range
        audio = audio_chunk.astype(np.float32)
        if np.max(audio) > 1.0:
            audio = audio / np.max(audio)
            
        perturbed_audio = audio.copy()
        
        if mode == "uap" and self.uap is not None:
            # Apply Universal Perturbation
            # UAP is usually 30s long. If chunk is shorter, handle it.
            if len(perturbed_audio) < self.uap.vector.shape[0]:
                # Pad with zeros or repeat logic (simplified here)
                perturbed_audio = self.uap.apply(perturbed_audio)
            else:
                perturbed_audio = self.uap.apply(perturbed_audio[:self.uap.vector.shape[0]])
        
        elif mode == "targeted":
            # Placeholder for targeted attack logic
            pass
            
        return perturbed_audio

    def transcribe(self, audio_array, mode="clean"):
        """
        Main transcription logic.
        """
        if audio_array is None:
            return "No audio input", 0, 0
            
        # Apply attack if needed
        processed_audio = self.apply_attack(audio_array, mode)
        
        # Save temp file for Whisper (Whisper requires file path or tensor, but tensor is faster)
        # We'll use the tensor interface if available, or file
        try:
            # Assuming transcribe_audio handles the tensor directly
            result = transcribe_audio(self.model, processed_audio, device=self.device)
            text = result['text']
            wer = result.get('wer', 0.0)
            snr = result.get('snr', 0.0)
        except Exception as e:
            print(f"Error during transcription: {e}")
            text = "Error processing audio."
            wer = 0.0
            snr = 0.0
            
        return text, wer, snr

# --- Gradio Interface Setup ---
def create_ui():
    transcriber = LiveTranscriber()
    
    with gr.Blocks(title="Adversarial ASR Demo") as demo:
        gr.Markdown("# Adversarial Speech Recognition")
        gr.Markdown("Select a mode and speak into the microphone.")
        
        with gr.Row():
            with gr.Column(scale=1):
                mode_select = gr.Radio(
                    choices=["clean", "uap", "targeted"],
                    value="clean",
                    label="Attack Mode",
                    info="Clean: No attack | UAP: Universal Perturbation | Targeted: CW Attack (Placeholder)"
                )
                
                mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Microphone Input"
                )
                
                submit_btn = gr.Button("Transcribe", variant="primary")
                
                with gr.Accordion("Transcript & Metrics"):
                    transcript_output = gr.Textbox(label="Transcript", lines=4)
                    metrics_output = gr.JSON(label="Metrics (WER, SNR)")
                    
                clear_btn = gr.Button("Clear Transcripts")
            
            with gr.Column(scale=1):
                status_box = gr.Textbox(label="System Status", value="Ready.")
                info_box = gr.Markdown(
                    """
                    ### Instructions
                    1. Select a mode.
                    2. Allow microphone access.
                    3. Speak clearly for ~3-5 seconds.
                    4. Click 'Transcribe' or press Enter.
                    
                    **Note:** UAP mode requires a pre-trained perturbation file.
                    """
                )
        
        # State variables
        state = {
            "history": []
        }

        def process_audio(audio_path):
            if audio_path is None:
                return "", {}, "Please provide audio."
            
            try:
                # Load audio
                # Note: Gradio returns a filepath or dict
                if isinstance(audio_path, dict):
                    if 'bytes' in audio_path:
                        # Handle byte stream if applicable
                        import io
                        with io.BytesIO(audio_path['bytes']) as f:
                            wav_data = sf.read(f)
                    elif 'name' in audio_path:
                        wav_data, _ = sf.read(audio_path['name'])
                    else:
                        return "", {}, "Unknown audio format."
                else:
                    wav_data, _ = sf.read(audio_path)
                
                # Resample to 16kHz if necessary
                if wav_data.shape[0] != SAMPLE_RATE:
                    print(f"Resampling from {wav_data.shape[0]} to {SAMPLE_RATE}")
                    # Simple resampling placeholder (implementation depends on library)
                    # For robustness, we assume input matches or use librosa/soundfile resampling
                
                # Get current mode
                current_mode = mode_select.value
                
                # Transcribe
                text, wer, snr = transcriber.transcribe(wav_data, current_mode)
                
                status = f"Processed. Mode: {current_mode}. WER: {wer:.2f}, SNR: {snr:.2f} dB"
                return text, {"WER": wer, "SNR": snr}, status
                
            except Exception as e:
                print(f"Process Error: {e}")
                return "", {}, f"Error: {e}"

        # Event Listeners
        submit_btn.click(
            fn=process_audio,
            inputs=[mic],
            outputs=[transcript_output, metrics_output, status_box]
        )
        
        clear_btn.click(
            fn=lambda: ("", {}, "Ready."),
            outputs=[transcript_output, metrics_output, status_box]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True)
