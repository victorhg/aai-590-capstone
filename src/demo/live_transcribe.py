import gradio as gr
import torch
import numpy as np
import whisper
import librosa
import soundfile as sf
import os
from src.attacks.uap import apply_universal_perturbation

class LiveTranscriptionUI:
    def __init__(self, model_name="base", uap_path="results/universal_perturbation_v.pt"):
        print("Loading Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"Model loaded on {self.device}.")
        
        # Load UAP if exists
        self.uap = None
        if os.path.exists(uap_path):
            print(f"Loading Universal Perturbation from {uap_path}...")
            try:
                self.uap = torch.load(uap_path)
                print("UAP loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load UAP. Error: {e}")
        else:
            print(f"Warning: UAP file not found at {uap_path}.")

    def transcribe(self, audio_input, mode="Clean"):
        """
        Transcribe audio. If mode is 'UAP', apply the perturbation before transcription.
        """
        if audio_input is None:
            return "Please upload or record audio first."

        try:
            # Whisper expects input in the format model takes (numpy array or file path)
            # We use numpy array for direct manipulation.
            
            # Mode 'Clean': Just pass to Whisper
            if mode == "Clean":
                result = self.model.transcribe(audio_input)
                return result["text"]
            
            # Mode 'UAP'
            elif mode == "Untargeted UAP":
                if self.uap is None:
                    return "Error: UAP not loaded. Check console logs."

                # Load audio with librosa for resampling and normalization (Whisper requirement)
                # Whisper handles the resampling, but we need the raw waveform
                y, sr = librosa.load(audio_input, sr=16000, mono=True)
                
                # Apply Perturbation
                # Ensure tensor format for model operations
                if isinstance(self.uap, np.ndarray):
                    self.uap = torch.from_numpy(self.uap)
                
                # Apply perturbation logic (assume epsilon clipping is handled inside attack module)
                perturbed_audio = apply_universal_perturbation(self.uap, y)
                
                # Whisper requires file path or numpy array. We save a temp file for Whisper
                # to handle the log_mel_spectrogram preprocessing correctly.
                temp_file = "temp_perturbed.wav"
                sf.write(temp_file, perturbed_audio, 16000)
                
                result = self.model.transcribe(temp_file)
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
                return result["text"]

            return "Unknown mode selected."

        except Exception as e:
            return f"Error during transcription: {str(e)}"

    def launch(self):
        with gr.Blocks(title="Whisper Adversarial Attack Demo") as demo:
            gr.Markdown("# **Whisper Adversarial Attack Demo**")
            gr.Markdown("### Live Transcription with UAP Injection")
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"], 
                        type="filepath", 
                        label="Input Audio"
                    )
                    
                    mode = gr.Radio(
                        choices=["Clean", "Untargeted UAP"], 
                        value="Clean", 
                        label="Attack Mode",
                        interactive=True
                    )
                    
                    transcribe_btn = gr.Button("Transcribe")
                    
                with gr.Column(scale=2):
                    output_text = gr.Textbox(label="Transcription Output", lines=5)

            gr.Markdown("**Instructions:**")
            gr.Markdown("1. Select a mode (Clean or Untargeted UAP).")
            gr.Markdown("2. Record your voice or upload an audio file.")
            gr.Markdown("3. Click 'Transcribe' to see the result.")
            gr.Markdown(f"4. **UAP Status:** {'Loaded' if self.uap is not None else 'Not Loaded'}")

            transcribe_btn.click(
                fn=self.transcribe,
                inputs=[audio_input, mode],
                outputs=output_text
            )

        demo.launch(share=False)

if __name__ == "__main__":
    # Configuration
    MODEL_SIZE = "base"
    UAP_FILE = "results/universal_perturbation_v.pt"
    
    app = LiveTranscriptionUI(model_name=MODEL_SIZE, uap_path=UAP_FILE)
    app.launch()
