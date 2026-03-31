# Demo Setup & Execution Guide

This guide provides instructions for setting up and running the Live Transcription Demonstration system. This project implements Universal Adversarial Perturbations (UAP) and Targeted Attacks against the OpenAI Whisper model.

## Prerequisites

Before running the live demo, ensure your environment is configured as per the main `README.md` and `requirements.txt`.

- **Python**: 3.10 or higher
- **GPU**: CUDA-enabled GPU (recommended for Whisper inference speed)
- **Libraries**: `torch`, `openai-whisper`, `gradio`, `sounddevice`, `numpy`

## Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2.  **Create Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    pip install gradio sounddevice queue pyaudio
    ```

4.  **Verify Audio Input**
    Run the following command to test your microphone permissions and volume:
    ```bash
    python -c "import sounddevice as sd; print(sd.query_devices())"
    ```

## Running the Live Transcription Demo

The system offers three modes of operation:

1.  **Clean Mode**: Baseline Whisper transcription (No attacks).
2.  **Untargeted UAP Mode**: Applies a pre-trained Universal Adversarial Perturbation. This corrupts the output transcript with high probability.
3.  **Targeted Attack Mode**: Attempts to inject a specific phrase into the user's speech (if trained).

### Execution

Navigate to the demo directory and run the application:

```bash
cd src/demo
python live_transcribe.py
```

### How to Use the UI

1.  Select the **Mode** from the dropdown:
    *   `Clean`: Standard Whisper output.
    *   `Untargeted UAP`: Use this if you have trained the UAP (see `notebooks/04_uap_training.ipynb`). The input audio is perturbed in real-time.
    *   `Targeted Attack`: Use this if you have trained a specific targeted attack (see `notebooks/08_train_demo_targeted_attack.ipynb`).
2.  **Start Transcription**: Click the **Start Transcription** button.
3.  **Speak**: Speak into your microphone.
4.  **Listen**: The system will process your audio and display the result below. You can adjust the volume/sensitivity settings if needed.
5.  **Stop**: Click **Stop Transcription** to end the session.

## Expected Behavior

*   **Latency**: You should expect a delay of roughly 2-5 seconds between speaking and seeing the result, depending on your CPU/GPU and audio buffer settings.
*   **Imperceptibility**: In Untargeted mode, the background noise (the UAP) should be inaudible to the human ear (target SNR > 35dB).
*   **Error**: If the demo crashes, ensure your `PATH` includes `ffmpeg` (required by Whisper for audio processing).

## Troubleshooting

### "PyAudio / PortAudio Error"
If you see an error regarding PyAudio or audio devices:
*   On Windows, ensure you have **Microsoft Visual C++ Redistributable** installed.
*   On Linux, try installing `libportaudio2` or reinstalling `portaudio`.

### "Whisper model loading..."
This indicates the model is downloading. The first run may take a minute to download the `base.en` or `small.en` model weights.

### Audio is too quiet or distorted
*   Check your OS volume levels.
*   Ensure the microphone input isn't set to "Do Not Disturb" in system settings.
*   If using a virtual audio cable (e.g., OBS), ensure the correct input device is selected in the Gradio interface.

### Demo does not show any text
*   Ensure you are speaking clearly.
*   Check the console logs for Whisper errors (e.g., silence detection).

## Advanced Usage

### Using Pre-Trained Attacks

If you want to use the trained perturbations without re-training them:

1.  **Untargeted**: Ensure `results/universal_perturbation_v.pt` exists (or the file path matches the script). The script is configured to load this automatically.
2.  **Targeted**: If you have trained a targeted perturbation, ensure `demo_assets/targeted_perturbation.pt` exists, and the file path in `src/demo/live_transcribe.py` is updated.

