
# AGENTS LEARNING & CONSTRAINTS

## Technical Constraints & "Gotchas"

### 1. Gradient Propagation Through Preprocessing
**Critical:** Whisper contains a `log_mel_spectrogram` preprocessing step. Standard PyTorch models might not propagate gradients through this if it's treated as a static transform.
*   **Learning:** The agent must ensure the input audio tensor `x` has `.requires_grad=True` *before* it is passed to the Mel-spectrogram function. The Mel function itself must be differentiable (PyTorch native implementation, not numpy/scipy).

### 2. Audio Sampling Rate Mismatches
**Critical:** Whisper is hardcoded for 16,000 Hz.
*   **Learning:** Any attack generated at 44.1kHz or 48kHz will be destroyed when downsampled to 16kHz for the model. Ideally, all attacks should be optimized and applied directly in the 16kHz domain to ensure the perturbation $delta$ aligns perfectly with the model inputs.

### 3. VRAM Usage & Batching
**Critical:** ASR models are memory intensive (Transformers). Calculating gradients adds significant overhead (activations must be stored).
*   **Learning:** Agents should default to `batch_size=1` for optimization loops. Doing large batch adversarial training might OOM (Out of Memory) consumer GPUs. Universal Perturbation training might need "Gradient Accumulation" strategies if batches are required, or simple iterative updates (one sample at a time).

### 4. Loss Function Selection
**Critical:** The PRD mentions `CTC Loss`. Whisper is an Encoder-Decoder model, but minimizing the log-likelihood of the correct transcription is the standard "untargeted" approach.
*   **Learning:**
    *   **Untargeted:** `Maximize Loss(y_true)`. Simply making the model *uncertain* is often enough.
    *   **Targeted:** `Minimize Loss(y_target)`. This is much harder.
    *   The agent needs to verify if it uses `SpeechBrain`'s CTC loss or simply taps into Whisper's internal cross-entropy loss. Accessing Whisper's specific loss requires looking at its `forward()` method signature.

### 5. Imperceptibility vs. Clipping
**Critical:** Audio is digital data, usually bounded `[-1.0, 1.0]`.
*   **Learning:** Simply adding noise $\delta$ can push audio out of valid range (clipping). This creates distinct "crackling" artifacts easiest for humans to hear.
*   **Constraint:** The attack loop *must* include a `clamp(min=-1, max=1)` step in the optimization or final output generation. The $L_\infty$ constraint (epsilon) also helps prevent this, but explicit clamping is safer.

### 6. Universal Perturbation Input Lengths
**Critical:** Universal perturbations ($v$) are usually fixed length (e.g., 2 seconds, 30 seconds). Input audio ($x$) varies in length.
*   **Learning:**
    *   If Audio < Perturbation: Crop the perturbation?
    *   If Audio > Perturbation: Tile/Repeat the perturbation?
    *   Review the PRD/Papers (Olivier 2023) to see the standard approach. Usually, UAPs are trained on fixed-length segments (e.g., the first 30s) or tiled.

## Project Structure Guidelines

### Modularity
*   **Separation:** Keep `attack_loop` separate from `data_loader`. This allows swapping "PGD" for "CW" or "UAP" without rewriting the audio fetching logic.
*   **Config:** Use arguments or config classes for `epsilon`, `learning_rate`, `iterations`. Hardcoding these makes the Week 3 "tuning" phase very painful.

## Reproducibility
*   **Seeding:** ASR results can be non-deterministic. Attacks are sensitive to initialization.
*   **Learning:** Agent must enforce `torch.manual_seed()` and `np.random.seed()` at the start of every experiment script.