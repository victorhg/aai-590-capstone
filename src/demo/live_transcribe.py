"""
Live Transcription System with Adversarial Capabilities
"""
import gradio as gr
import torch
import numpy as np
from queue import Queue
import sounddevice as sd
import soundfile as sf
import librosa
import time
import threading

# Local imports
from src.models.whisper_wrapper import load_whisper_model
from src.demo.audio_stream import AudioStreamManager
from src.demo.attack_utils import UAPManager

# Global State
MODES = ["Clean", "Untargeted UAP"]
MODE_CLEAN = 0
MODE_UAP = 1

class LiveTranscriber:
    def __init__(self):
        print("Initializing Live Transcriber...")
        
        # Load Whisper Model
        # Using 'base' for faster inference on CPU/CPU+GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = load_whisper_model(device=self.device)
        
        # Initialize Attack Manager
        # Attempting to load the UAP trained in notebook 04
        uap_path = "results/universal_perturbation_v_80.pt"
        try:
            self.uap_manager = UAPManager(uap_path=uap_path)
        except Exception as e:
            print(f"Warning: Could not load UAP manager. Attacks will be disabled. Error: {e}")
            self.uap_manager = None

        # Audio Stream Setup
        self.sample_rate = 16000
        self.stream = None
        self.transcription_queue = Queue()
        
        # State
        self.is_recording = False
        self.current_mode = MODE_CLEAN

    def start_recording(self):
        """Start the audio recording loop."""
        if self.is_recording:
            return "Already recording!"
            
        self.is_recording = True
        self.audio_buffer = []
        
        # Callback for audio chunks
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio stream status: {status}")
            self.audio_buffer.append(indata.copy())
            
        try:
            # Start stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1, # Mono
                callback=callback,
                blocksize=2048
            )
            self.stream.start()
            print("Recording started...")
            
            # Start processing thread
            threading.Thread(target=self.process_audio_loop, daemon=True).start()
            
        except Exception as e:
            print(f"Error starting stream: {e}")
            self.is_recording = False

    def process_audio_loop(self):
        """Process audio chunks while recording."""
        while self.is_recording:
            if len(self.audio_buffer) > 0:
                # Get the last chunk
                chunk = self.audio_buffer.pop(0)
                
                # Apply Adversarial Attack if in UAP mode
                if self.current_mode == MODE_UAP and self.uap_manager:
                    # UAP expects [channels, samples] or [samples], chunk is usually [frames, channels] or [frames]
                    # Sounddevice returns shape (N, channels)
                    if chunk.ndim == 1:
                        chunk = chunk.unsqueeze(1)
                    
                    perturbed_chunk = self.uap_manager.apply_uap(chunk)
                    if perturbed_chunk is not None:
                        # Convert back to [samples] for whisper
                        audio_input = perturbed_chunk.squeeze(1).cpu().numpy()
                    else:
                        audio_input = chunk.squeeze(1).cpu().numpy()
                else:
                    # Clean mode
                    audio_input = chunk.squeeze(1).cpu().numpy()
                
                # Transcribe
                try:
                    result = self.model.transcribe(
                        audio_input,
                        language='en',
                        fp16=False if self.device == 'cpu' else True
                    )
                    text = result["text"].strip()
                    
                    # Calculate SNR if applicable
                    snr_info = ""
                    if self.current_mode == MODE_UAP and self.uap_manager:
                        # Reconstruct original for SNR (approximate)
                        original = chunk.squeeze(1).cpu().numpy()
                        perturbed = audio_input
                        snr = self.uap_manager.get_snr(
                            torch.tensor(original, dtype=torch.float32),
                            torch.tensor(perturbed, dtype=torch.float32)
                        )
                        snr_info = f" (SNR: {snr:.2f}dB)"
                    
                    self.transcription_queue.put(text + snr_info)
                    
                except Exception as e:
                    print(f"Transcription error: {e}")
                    
            time.sleep(0.01) # Small sleep to prevent busy waiting

    def stop_recording(self):
        """Stop recording and get result."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        return "Stopped."

    def set_mode(self, mode_name):
        """Update the attack mode."""
        if mode_name == "Clean":
            self.current_mode = MODE_CLEAN
        elif mode_name == "Untargeted UAP":
            self.current_mode = MODE_UAP
        return f"Mode changed to: {mode_name}"

# Gradio UI Setup
def setup_ui():
    transcriber = LiveTranscriber()
    
    with gr.Blocks(title="Whisper Adversarial Demo") as demo:
        gr.Markdown("# Whisper Live Transcription: Clean vs. Adversarial")
        
        with gr.Row():
            with gr.Column():
                status_text = gr.Textbox(label="Status")
                mode_select = gr.Dropdown(choices=MODES, value="Clean", label="Operation Mode")
                record_btn = gr.Button("Start Recording", variant="primary")
                stop_btn = gr.Button("Stop Recording", variant="stop")
                
            with gr.Column():
                transcript_display = gr.Textbox(label="Live Transcript", lines=5)
        
        # Event Handlers
        record_btn.click(
            fn=lambda: status_text.update(value="Recording..."),
            inputs=None,
            outputs=status_text
        ).then(
            fn=transcriber.start_recording,
            inputs=None,
            outputs=None
        ).then(
            fn=lambda: status_text.update(value="Live Transcribing..."),
            inputs=None,
            outputs=status_text
        )
        
        stop_btn.click(
            fn=transcriber.stop_recording,
            inputs=None,
            outputs=status_text
        )
        
        # Live Update
        def update_transcript():
            if transcriber.is_recording:
                try:
                    new_text = transcriber.transcription_queue.get_nowait()
                except:
                    new_text = ""
                return transcript_display.update(value=new_text), status_text.update(value="Live Transcribing...")
            else:
                return transcript_display.update(value="Stop recording to see final text."), status_text.update(value="Idle")

        demo.queue()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    setup_ui()
