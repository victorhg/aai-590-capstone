import gradio as gr
import torch
import numpy as np
from src.models.whisper_wrapper import WhisperModelWrapper
from src.demo.audio_stream import AudioStream
from src.attacks.uap import UniversalAdversarialPerturbation

class LiveTranscriptionSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Whisper on {self.device}...")
        
        # Initialize Whisper Model
        self.model = WhisperModelWrapper(device=self.device)
        
        # Initialize UAP Handler (Placeholder for loading trained perturbations)
        # In a real scenario, we would load from 'results/universal_perturbation_v.pt'
        self.uap_handler = UniversalAdversarialPerturbation(
            perturbation_path=None, # Load dynamically in inference
            epsilon=0.3,
            device=self.device
        )
        
        self.audio_stream = AudioStream()
        self.is_recording = False
        self.current_audio_buffer = None

    def transcribe_audio(self, audio_data, mode):
        """
        Transcribe audio data based on the selected mode.
        """
        if mode == "Clean":
            # Apply no perturbation
            processed_audio = audio_data
        elif mode == "Untargeted UAP":
            # Apply Universal Adversarial Perturbation
            # Note: We assume the perturbation is trained for 30s segments matching the input shape
            if self.uap_handler.perturbation is None:
                print("Loading UAP vector...")
                # Attempt to load from results folder (standard location from training)
                try:
                    self.uap_handler.load_perturbation('results/universal_perturbation_v.pt')
                except Exception as e:
                    return f"Error: UAP not found or loaded. ({e})"

            # Apply UAP (Assuming shape compatibility, otherwise tile/repeat logic would go here)
            # We add the perturbation to the input
            processed_audio = self.uap_handler.apply(audio_data)
            
        elif mode == "Targeted Attack":
            # Placeholder for targeted injection logic
            return "Targeted attack mode not implemented in this demo script yet."
            
        else:
            return "Invalid mode selected."

        # Normalize audio to [-1, 1]
        if np.abs(processed_audio).max() > 1.0:
            processed_audio = processed_audio / np.abs(processed_audio).max()

        # Transcribe
        try:
            result = self.model.transcribe(processed_audio)
            return result
        except Exception as e:
            return f"Transcription Error: {e}"

    def toggle_recording(self, status_text, mode_choice):
        """Handles the recording state and processes audio."""
        if not self.is_recording:
            # Start Recording
            print(f"Starting audio stream in mode: {mode_choice}")
            self.audio_stream.start()
            self.is_recording = True
            return "Recording... (30s)", status_text
        else:
            # Stop Recording & Process
            print("Stopping audio stream and processing...")
            self.audio_stream.stop()
            self.is_recording = False
            
            # Capture 30 seconds of audio (or available buffer)
            # Note: AudioStream logic handles buffering, we need to get the data here
            try:
                # Get the last N samples (assuming 30s target based on Whisper default)
                raw_audio = self.audio_stream.get_buffer()
                
                # Normalize to float32 [-1, 1]
                if raw_audio.dtype != np.float32:
                    raw_audio = raw_audio.astype(np.float32)
                if raw_audio.max() > 1.0 or raw_audio.min() < -1.0:
                    raw_audio = raw_audio / np.abs(raw_audio).max()
                
                transcript = self.transcribe_audio(raw_audio, mode_choice)
                
                # Update status with metrics
                status = f"Done. Transcript: {transcript}"
                return "Ready", status
                
            except Exception as e:
                return f"Error: {e}", status_text

def main():
    system = LiveTranscriptionSystem()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Live Speech Recognition Demo")
        gr.Markdown("Select an attack mode and click Start to record a 30-second segment.")
        
        with gr.Row():
            with gr.Column():
                mode = gr.Radio(
                    choices=["Clean", "Untargeted UAP", "Targeted Attack"],
                    value="Clean",
                    label="Attack Mode"
                )
                record_btn = gr.Button("Start / Stop Recording")
                status_display = gr.Textbox(label="Status", interactive=False, value="Ready")
                transcript_display = gr.Textbox(label="Transcription", interactive=False, lines=5)
                
                with gr.Accordion("Metrics", open=False):
                    snr_display = gr.Textbox(label="Est. SNR (dB)", interactive=False)
                    wer_display = gr.Textbox(label="Est. WER", interactive=False)
            
            with gr.Column():
                # Optional: Visual indicator placeholder
                gr.Markdown("**Visual Feedback**")
                indicator = gr.Image(label="Audio Visualization", placeholder="Waiting for audio...")
        
        record_btn.click(
            fn=lambda status, mode: system.toggle_recording(status, mode),
            inputs=[status_display, mode],
            outputs=[status_display, transcript_display]
        )
        
        # Handle SNR/WER updates (Simulated or real-time hooks)
        # For this demo, we'll just print to console or update the status line with basic feedback
        # In a full implementation, these would update live.

    if __name__ == "__main__":
        demo.launch(share=False)

