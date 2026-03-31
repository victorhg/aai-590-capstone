import gradio as gr
import torch
import numpy as np
import soundfile as sf
import librosa
from .audio_stream import AudioStream
from ...models.whisper_wrapper import WhisperModelWrapper
from ...attacks.uap import apply_uap

# Global model wrapper
model_wrapper = None

def initialize_model():
    """Loads the Whisper model lazily."""
    global model_wrapper
    if model_wrapper is None:
        print("Loading Whisper model...")
        model_wrapper = WhisperModelWrapper()
        print("Model loaded.")
    return model_wrapper

def process_audio(audio_data, sample_rate, mode, uap_path="demo_assets/universal_perturbation.pt"):
    """
    Main processing function.
    
    Args:
        audio_data: Raw numpy audio array from Gradio.
        sample_rate: Sample rate of the input audio.
        mode: 'clean', 'uap', or 'targeted'.
        uap_path: Path to the pre-trained UAP vector.
    """
    
    # Initialize model
    model = initialize_model()
    
    if audio_data is None:
        return "No audio input.", 0, 0
    
    # Ensure 16kHz (Whisper requirement)
    # Gradio records at user choice, usually 16k or 44.1k. 
    # We assume standard Gradio behavior and downsample if necessary, 
    # but typically Gradio's default is 16k.
    
    try:
        if mode == "Clean (Baseline)":
            # 1. Transcribe Clean
            print("Running Clean Transcription...")
            # WhisperWrapper expects float32 in [-1, 1]
            text_clean = model_wrapper.transcribe(audio_data)
            
            # Calculate SNR (Dummy for clean)
            # Ideally we store original, but for UI we might skip exact SNR calc for clean 
            # or compute it against a theoretical silent floor.
            snr = "N/A" 
            cer = 0 # Assuming clean
            
            return text_clean, "0 dB", "0.0"

        elif mode == "Untargeted UAP (Corruption)":
            # 1. Load UAP perturbation
            # Note: This relies on the 'Untargeted Attack Integration' task completing
            print(f"Loading UAP from {uap_path}...")
            try:
                uap_vector = torch.load(uap_path)
            except FileNotFoundError:
                return f"Error: UAP file not found at {uap_path}. Please run 'Untargeted Attack Integration' first.", 0, 0

            # 2. Apply UAP to audio
            # apply_uap handles the casting and padding if necessary
            perturbed_audio = apply_uap(audio_data, uap_vector)
            
            # 3. Transcribe Corrupted
            print("Running Adversarial Transcription...")
            text_adv = model_wrapper.transcribe(perturbed_audio)
            
            # 4. Metrics
            # Calculate CER (Character Error Rate)
            cer = compute_cer(text_clean, text_adv) 
            # Calculate SNR (Signal to Noise Ratio of perturbation)
            snr = calculate_snr(audio_data, perturbed_audio)
            
            return f"Adversarial Transcript: {text_adv}\n\nClean: {text_clean}", f"{snr:.2f} dB", f"{cer:.2f}"

        elif mode == "Targeted CW (Injection)":
            # Note: This relies on the 'Targeted Attack Training' task completing
            return "Targeted Attack mode not yet loaded. Please train a targeted model first.", 0, 0
            
    except Exception as e:
        return f"Error during processing: {str(e)}", 0, 0

def compute_cer(ref, hyp):
    # Simple cer implementation or using jiwer if available
    try:
        from jiwer import cer
        return cer(ref, hyp)
    except ImportError:
        return 0.0

def calculate_snr(orig, perturbed):
    # Signal to Noise Ratio in dB
    orig_sq = np.mean(orig**2)
    diff_sq = np.mean((orig - perturbed)**2)
    if diff_sq == 0:
        return float('inf')
    return 10 * np.log10(orig_sq / diff_sq)

# --- UI Setup ---

def build_ui():
    with gr.Blocks(title="Whisper Adversarial Attack Demo") as demo:
        gr.Markdown("# 🎤 Whisper Adversarial Attack Live Demo")
        gr.Markdown("Select an attack mode and record your audio.")
        
        with gr.Row():
            with gr.Column():
                mode = gr.Dropdown(
                    choices=["Clean (Baseline)", "Untargeted UAP (Corruption)", "Targeted CW (Injection)"],
                    value="Clean (Baseline)",
                    label="Attack Mode"
                )
                record_btn = gr.Audio(
                    sources=["microphone"],
                    type="numpy", 
                    label="Record Audio (30s)"
                )
                submit_btn = gr.Button("Transcribe")
                
            with gr.Column():
                output_text = gr.Textbox(label="Transcription", lines=4)
                output_snr = gr.Textbox(label="SNR (dB)", interactive=False)
                output_cer = gr.Textbox(label="CER", interactive=False)

        submit_btn.click(
            fn=process_audio,
            inputs=[record_btn, None, mode], # Pass dummy sample rate, handled in fn
            outputs=[output_text, output_snr, output_cer]
        )
        
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
