# Technical Report: Universal Adversarial Perturbations for OpenAI Whisper ASR

## 1. Project Overview
This project explores the vulnerability of OpenAI's Whisper Automatic Speech Recognition (ASR) system to Universal Adversarial Perturbations (UAP). Unlike targeted single-sample attacks, UAPs are designed to remain consistent across a distribution of input audio, causing the model to fail globally with a small, imperceptible noise injection.

**Objective:**
- Demonstrate the feasibility of bypassing Whisper's transcription capabilities using UAPs.
- Implement a robust defense mechanism (Randomized Smoothing) against such attacks.
- Evaluate the trade-off between audio integrity (SNR) and classification accuracy (WER/CER).

## 2. System Architecture

The system is built around the OpenAI Whisper model (e.g., `base` or `small` variant) and PyTorch for tensor operations.

### 2.1. Core Components
1.  **Preprocessing Layer**: Whisper uses a `log_mel_spectrogram` function. Unlike standard CNNs where inputs are fixed tensors, Whisper performs preprocessing on the raw audio tensor *inside* the model architecture (specifically in the `forward` pass or `encoder` input).
2.  **Attack Vector Layer**: 
    - *Single-Sample PGD*: Iteratively updates a perturbation vector for a specific input to maximize loss (minimize likelihood of correct transcription).
    - *Universal Perturbation*: Trains a single vector $v$ that is added to any input in the training set to maximize loss across the dataset.
3.  **Defense Layer**: Randomized Smoothing. By adding Gaussian noise during inference and taking the majority vote, the model becomes robust to small adversarial perturbations.
4.  **Evaluation Layer**: Computes Word Error Rate (WER), Character Error Rate (CER), and Signal-to-Noise Ratio (SNR) to quantify attack success and imperceptibility.

## 3. Key Technical Decisions

### 3.1. Sampling Rate Lock (16kHz)
*   **Decision**: All audio inputs are strictly resampled to 16,000 Hz.
*   **Rationale**: Whisper is hardcoded to expect 16kHz audio. Generating attacks at higher rates (44.1kHz/48kHz) results in misaligned perturbations after downsampling, rendering the attack ineffective.

### 3.2. Gradient Propagation Through Preprocessing
*   **Decision**: Ensure the raw audio tensor has `.requires_grad=True` *before* passing it to the Whisper model.
*   **Rationale**: Standard implementations often wrap inputs in a `torch.nn.Functional` or standard transform which might break the autograd graph. The `log_mel_spectrogram` function must remain differentiable (PyTorch native) for gradients to flow back to the input noise.

### 3.3. Input Clamping (Imperceptibility)
*   **Decision**: Explicitly clamp the final perturbed audio to the range `[-1.0, 1.0]` after every optimization iteration and after generation.
*   **Rationale**: Audio data is float32 in `[-1, 1]`. Accumulating gradient steps without clamping can drive the signal out of valid bounds, creating "clipping" artifacts which are the most easily perceptible and damaging to the signal.

### 3.4. VRAM Optimization (Batch Size)
*   **Decision**: Use `batch_size=1` for all optimization loops (PGD and UAP training).
*   **Rationale**: ASR models store significant activation memory (transformer layers). Gradient accumulation adds overhead. Batching more than one sample at a time risks OOM (Out of Memory) errors on consumer GPUs.

### 3.5. Universal Perturbation Handling
*   **Decision**: Use "Accumulated Gradient" approach where the global vector $v$ is updated as $v \leftarrow v - \text{learning\_rate} \times \sum \nabla L$.
*   **Rationale**: This is the most efficient method for UAP training as it avoids storing gradients for every sample in memory, allowing the vector to evolve across the dataset without consuming excessive VRAM.

## 4. Implementation Steps

### Week 1: Foundation & EDA
- **Data Acquisition**: Downloaded LibriSpeech `test-clean` and CommonVoice datasets.
- **Preprocessing**: Implemented `audio_loader.py` to ensure 16kHz resampling, float32 normalization, and valid audio range checks.
- **Analysis**: Created interactive analysis notebooks to visualize waveforms and Mel-spectrograms, validating the dataset structure.

### Week 2: Baseline & PGD
- **Baseline**: Evaluated Whisper on clean data to establish a WER/CER baseline.
- **PGD Attack**: Implemented a generic PGD attack in `src/attacks/pgd.py`. The attack optimizes a perturbation vector $\delta$ by maximizing the model's loss, effectively making the model "confused" about the input.

### Week 4-5: UAP Training
- **Training Loop**: Implemented a training loop that iterates through the dataset. For each sample, it calculates the gradient of the loss w.r.t the input and accumulates it to update the global perturbation $v$.
- **Projection**: Periodically projected $v$ back into the $\epsilon$-ball to ensure the perturbation remains small and within the L-infinity constraint.
- **Validation**: Split data into Train/Validation sets to monitor the "Success Rate" (high CER indicates attack success).

### Week 5-6: Defense
- **Randomized Smoothing**: Implemented a defense strategy where input is perturbed with Gaussian noise during inference.
- **Evaluation**: Tested if the UAP attack could fool the smoothed model. The report documents the drop in Attack Success Rate versus the increase in Clean WER due to noise.

## 5. Results Summary
- **Success Rate**: The UAP achieved a significant success rate (defined here as CER > 0.5) on the test set, demonstrating that a single global noise pattern can degrade performance across the entire distribution.
- **Imperceptibility**: The perturbations maintained a high Signal-to-Noise Ratio (SNR), confirming they were audible but not obviously distorted to the human ear.
- **Defense Effectiveness**: Randomized Smoothing provided a layer of robustness, raising the threshold required for the attack to succeed.

## 6. Future Work
- **Targeted Attacks**: Implement Carlini-Wagner (CW) attacks for specific, controlled output modifications.
- **Hardware Optimization**: Explore mixed-precision training (FP16) to reduce VRAM requirements for larger batch sizes.
- **Cross-Domain Evaluation**: Test UAPs trained on LibriSpeech against CommonVoice to verify true universality across speaker accents and languages.
