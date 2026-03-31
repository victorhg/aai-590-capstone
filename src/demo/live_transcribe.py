"""
Live Transcription Demo System
Supports Clean, Untargeted UAP, and Targeted Attack modes.
"""
import gradio as gr
import torch
import numpy as np
from src.models.whisper_wrapper import WhisperASR
from src.demo.audio_stream import AudioStream
from src.demo.attack_utils import (
    apply_untargeted_uap,
    apply_targeted_attack
)

class LiveTranscribeApp:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.asr = WhisperASR(device=device)
        self.audio_stream = AudioStream()
        
        # Load perturbations
        # Try loading UAP
        self.uap = None
        try:
            self.uap = torch.load('results/universal_perturbation_v.pt', map_location=device)
            print("Loaded Untargeted UAP successfully.")
        except:
            print("Untargeted UAP not found.")
            
        # Try loading Targeted Perturbation
        self.targeted_pert = None
        try:
            self.targeted_pert = torch.load('demo_assets/targeted_perturbation.pt', map_location=device)
            print("Loaded Targeted Perturbation successfully.")
        except FileNotFoundError:
            print("Targeted Perturbation not found in demo_assets/.")

    def process_audio(self, audio_data, mode):
        """
        Main inference loop.
        audio_data: numpy array
        mode: "Clean", "Untargeted", "Targeted"
        """
        if mode == "Clean":
            # Standard Whisper
            transcript = self.asr.transcribe(audio_data)
            metrics = {"SNR": "N/A", "WER": "N/A"}
            attack_status = "No Attack"
            
        elif mode == "Untargeted":
            # Apply UAP
            if self.uap is not None:
                audio_perturbed = apply_untargeted_uap(audio_data, self.uap)
                transcript = self.asr.transcribe(audio_perturbed)
                # Calculate simple SNR for demo
                snr_val = self._calculate_snr(audio_data, audio_perturbed)
                metrics = {"SNR": f"{snr_val:.2f}dB", "WER": "N/A"}
                attack_status = "UAP Applied"
            else:
                transcript = "Error: UAP model not loaded."
                metrics = {"SNR": "Error", "WER": "Error"}
                attack_status = "UAP Not Found"

        elif mode == "Targeted":
            # Apply Targeted Attack
            if self.targeted_pert is not None:
                audio_perturbed = apply_targeted_attack(audio_data, self.targeted_pert)
                transcript = self.asr.transcribe(audio_perturbed)
                # Calculate SNR
                snr_val = self._calculate_snr(audio_data, audio_perturbed)
                metrics = {"SNR": f"{snr_val:.2f}dB", "WER": "N/A"}
                attack_status = "Targeted Attack Applied"
            else:
                transcript = "Error: Targeted Perturbation not found. Please run training notebook first."
                metrics = {"SNR": "Error", "WER": "Error"}
                attack_status = "Model Not Found"
        else:
            transcript = "Unknown mode"
            metrics = {"SNR": "N/A", "WER": "N/A"}
            attack_status = "Unknown"

        return transcript, metrics, attack_status

    def _calculate_snr(self, clean_audio, noisy_audio):
        """Calculate Signal-to-Noise Ratio in dB."""
        # Handle different shapes/lengths by taking min or center
        min_len = min(len(clean_audio), len(noisy_audio))
        clean = clean_audio[:min_len]
        noisy = noisy_audio[:min_len]
        
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean((clean - noisy) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)

def create_demo_ui():
    app = LiveTranscribeApp()
    
    with gr.Blocks(title="Adversarial Audio Transcription Demo") as demo:
        gr.Markdown("# Adversarial Audio Transcription Demo")
        gr.Markdown("Select a mode to see how attacks affect Whisper transcriptions.")
        
        with gr.Row():
            with gr.Column():
                mode_select = gr.Radio(
                    choices=["Clean", "Untargeted", "Targeted"],
                    value="Clean",
                    label="Attack Mode"
                )
                record_btn = gr.Button("🎤 Start Recording (30s)")
                stop_btn = gr.Button("⏹ Stop Recording")
            
            with gr.Column():
                transcript_output = gr.Textbox(label="Transcription", lines=5)
                metrics_output = gr.JSON(label="Metrics (SNR)")
                status_output = gr.Textbox(label="Status", value="Ready")
        
        # State variables
        is_recording = gr.State(False)
        
        def start_recording():
            is_recording = True
            return is_recording, "Recording...", "Active"
        
        def stop_recording():
            # Capture audio
            audio_data = app.audio_stream.get_audio()
            # Process
            mode = mode_select.value
            transcript, metrics, status = app.process_audio(audio_data, mode)
            is_recording = False
            return is_recording, "", transcript, metrics, status

        record_btn.click(start_recording, inputs=[], outputs=[is_recording, status_output, status_output])
        stop_btn.click(stop_recording, inputs=[], outputs=[is_recording, record_btn, transcript_output, metrics_output, status_output])

    return demo

if __name__ == "__main__":
    ui = create_demo_ui()
    ui.launch()
