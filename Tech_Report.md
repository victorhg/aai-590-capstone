# Audio Adversarial Attacks on Whisper

AAI-590 Capstone Project
Victor Hugo Germano
Shiley-Marcos School of Engineering, University of San Diego

---

## 1. Executive Summary

This project investigates whether OpenAI Whisper can be fooled by small, carefully optimized perturbations to audio that humans may barely notice but that significantly degrade or redirect automatic speech recognition output. The capstone implemented four attack families against Whisper-base in a digital white-box setting:

- **Untargeted Projected Gradient Descent (PGD)** — creates a custom perturbation for one audio clip at a time, achieving a 92% fooling rate across 50 test samples at 26.58 dB SNR.
- **Universal Adversarial Perturbation (UAP)** — learns a single reusable perturbation that degrades Whisper's transcription across many audio clips, achieving a 60% fooling rate on held-out samples.
- **Targeted Carlini-Wagner (CW)** — forces Whisper to produce a specific phrase on a per-sample basis, with demonstrated success on a 5-sample evaluation batch.
- **Universal Carlini-Wagner (UCW)** — combines CW's targeted objective with UAP's universal strategy, training a single perturbation that forces a target phrase across any input, achieving 58.6% success on a 350-sample held-out validation set.

The main finding is that Whisper remains vulnerable across all four attack families. Untargeted attacks substantially change transcription output; the universal untargeted perturbation generalizes to unseen utterances; targeted per-sample attacks achieve phrase injection; and the universal targeted perturbation achieves majority success on held-out validation data. At the same time, the most effective attacks operate at SNR levels that are more audible than ideal, so effectiveness and imperceptibility remain the central tradeoff.

---

## 2. Problem and Motivation

Automatic speech recognition is increasingly deployed in voice assistants, transcription systems, accessibility tools, and safety-critical interfaces. That deployment makes adversarial robustness a practical security question. If an attacker can add a carefully crafted perturbation $\delta$ to clean audio $x$, the ASR model may produce a degraded or entirely different transcription:

$$
x_{adv} = \operatorname{clip}(x + \delta,\ -1,\ 1)
$$

This project tests that vulnerability on Whisper and compares four attack modes:

1. **Untargeted degradation** — make Whisper transcribe audio incorrectly.
2. **Universal untargeted degradation** — learn one perturbation that works across many inputs.
3. **Targeted injection** — force Whisper to emit a specific phrase.
4. **Universal targeted injection** — force a specific phrase across many inputs with a single perturbation.

Understanding these vulnerabilities is critical because it enables security applications that prevent malicious actors from bypassing voice authentication, supports privacy protection by clarifying how to safeguard sensitive speech data, drives robustness improvement through the development of more resilient ASR models, and addresses safety considerations for voice-controlled systems that cannot be hijacked through audio manipulation (Olivier & Raj, 2022).

This follows the threat model described by Olivier and Raj (2023) for Whisper-specific attacks and by Neekhara et al. (2019) for universal perturbations in ASR.

---

## 3. Dataset and Experimental Setup

### 3.1 Primary Dataset

The core dataset is **LibriSpeech test-clean**, stored locally under `data/LibriSpeech`. It provides clean English read speech from public-domain audiobooks in the LibriVox project, with aligned transcripts. The corpus is distributed under a Creative Commons license and is recognized for its diverse speakers and high-quality recordings (Airlangga, 2023).

| Property | Value |
|---|---|
| Total audio files | 2,620 |
| Mean duration | 7.42 seconds |
| Maximum duration | 34.95 seconds |
| Sample rate | 16 kHz (Whisper requirement) |
| Amplitude range | Normalized to $[-1.0, 1.0]$ |

The dataset is organized into subsets that differ in recording difficulty and alignment quality. The test-clean partition was chosen to ensure diverse evaluation across accents, intonations, and speech patterns while maintaining high recording quality for controlled experimentation.

### 3.2 Evaluation Subsets

Different experiments use different subsets depending on computational requirements:

| Experiment | Training Samples | Validation Samples | Speakers |
|---|---:|---:|---|
| Clean baseline (Notebook 02) | — | 50 | Mixed |
| PGD attack (Notebook 03) | — | 50 | Mixed |
| UAP training (Notebook 04) | ~100 (10% of dataset) | 20 | Mixed |
| CW targeted (Notebook 05) | — | 5 | Mixed |
| UCW training (Notebook 06) | 1,600 | 350 | 40 speakers each |

The UCW train/validation split is saved in `results/ucw_split_manifest.json` for reproducibility.

### 3.3 Baseline Caveats

Whisper inserts punctuation and slightly normalizes words relative to LibriSpeech ground-truth transcripts, which can inflate baseline WER even on clean audio. Text normalization (lowercase, remove punctuation, strip whitespace) is applied before computing metrics to mitigate this effect.

The baseline evaluation notebook uses `openai/whisper-base` via the `WhisperASRWithAttack` wrapper. The clean baseline on 50 samples:

| Metric | Value |
|---|---:|
| Mean WER | 4.78% |
| Mean CER | 1.98% |
| Samples evaluated | 50 |

Representative example from `results/baseline_clean_results.json`:

> **Ground truth:** "he sat down weak bewildered and one thought was uppermost zora"
> **Whisper output:** "he sat down weak bewildered and one thought was uppermost sora"
> WER: 9.1% | CER: 1.6%

---

## 4. High-Level ML Pipeline

The project is an adversarial ML pipeline built around a frozen Whisper model. The key architectural point is that optimization happens in the **raw waveform domain**, while the decision boundary being attacked is the **log-mel spectrogram** representation Whisper consumes internally.

```
Audio File (.flac)
    │
    ▼
load_audio_tensor() ── resample to 16 kHz, normalize to [-1, 1]
    │
    ▼
Attack (PGD / UAP / CW / UCW) ── adds perturbation δ
    │
    ▼
_preprocess_audio() ── pad or crop to 480,000 samples (30 s)
    │
    ▼
_compute_mel_spectrogram() ── STFT → power spectrum → mel projection → log-scale
    │
    ▼
Whisper encoder-decoder ── frozen weights, gradient flows back to δ
    │
    ▼
Loss computation ── backprop to update δ
    │
    ▼
Evaluation ── WER, CER, SNR, success rate
```

Performing optimization in the waveform domain ensures the resulting adversarial audio remains a valid playable signal. Variable-length utterances are handled through tiling or cropping: audio shorter than the perturbation is padded, and audio longer than the perturbation has the perturbation tiled across it. Explicit clamping to $[-1, 1]$ after perturbation prevents clipping artifacts.

---

## 5. Approach and Algorithms

### 5.1 Whisper Attack Wrapper

The project uses a custom differentiable wrapper around HuggingFace `openai/whisper-base` implemented in `src/models/whisper_wrapper.py`. Instead of treating preprocessing as a black box, the code manually computes the full mel-spectrogram pipeline in PyTorch to preserve gradient flow:

1. **STFT** — Hann window, $n_{\text{fft}} = 400$, $\text{hop\_length} = 160$
2. **Power spectrum** — $|\text{STFT}|^2$, drop last frame to produce exactly 3,000 frames
3. **Mel projection** — multiply by pre-extracted mel filters (80 bins) from `WhisperFeatureExtractor`
4. **Log-scale normalization** — Whisper's standard log-mel normalization:

$$
\text{log\_mels} = \frac{\log_{10}(\max(\text{mels},\ 10^{-10})) + 4.0}{4.0}
$$

This normalization amplifies the relative effect of perturbations in low-energy spectral regions, making frequency gaps between formants and unvoiced intervals particularly high-leverage attack surfaces.

All Whisper weights are frozen (`requires_grad=False`); only the adversarial perturbation tensor $\delta$ is optimized. The wrapper provides two loss modes:

- **Untargeted**: Minimize the standard deviation of encoder hidden states to encourage incoherent representations: $L = -\mathbb{E}[\text{std}(h)]$
- **Targeted**: Cross-entropy loss with Whisper's full decoder prefix (`<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, `<|notimestamps|>`) followed by tokenized target text, labels ignore the prefix positions.

### 5.2 Untargeted PGD

PGD is implemented in `src/attacks/pgd.py`. It iteratively updates adversarial audio with signed gradient steps and projects the result onto the $L_\infty$ perturbation bound:

$$
\delta_{t+1} = \Pi_{\|\delta\|_\infty \le \epsilon}\!\left(\delta_t + \alpha \cdot \operatorname{sign}(\nabla_\delta L)\right)
$$

After projection onto the $\epsilon$-ball, the perturbed audio is also clamped to the valid waveform range $[-1, 1]$. The attack starts from random initialization within the $\epsilon$-ball.

| Parameter | Value |
|---|---|
| $\epsilon$ (L$_\infty$ bound) | 0.01 |
| $\alpha$ (step size) | 0.001 |
| Iterations | 20 |
| Batch size | 1 |
| Initialization | Random within $\epsilon$-ball |

PGD is sample-specific (one perturbation per input), making it suitable for per-utterance evaluation and directly comparable to published Whisper attack literature.

### 5.3 Universal Adversarial Perturbation (UAP)

The UAP method is implemented in `src/attacks/uap.py` with shared helpers in `src/attacks/base.py` and `src/attacks/utils.py`. A single perturbation tensor $v$ of shape $[1, 80000]$ (5 seconds at 16 kHz) is trained across a dataset of utterances.

The training loop:

1. Initializes $v$ as zeros
2. Iterates through a training set of ~100 utterances for 20 epochs
3. Skips examples already fooled by the current perturbation (concentrating optimization on the hardest samples)
4. Updates $v$ using gradient-based optimization
5. Projects $v$ back into the allowed $L_\infty$ ball after each update

| Parameter | Value |
|---|---|
| UAP length | 80,000 samples (5.0 seconds) |
| $\epsilon$ (L$_\infty$ bound) | 0.08 |
| Epochs | 20 |
| Training samples | ~100 (10% of dataset) |

Length mismatches between $v$ and input audio are handled by `tile_to_length()` in `src/attacks/utils.py`: if the audio is shorter than $v$, the perturbation is cropped; if longer, it is tiled (repeated) to cover the full duration.

The UAP is the most operationally interesting attack because it is reusable. A pre-computed perturbation can act as a "digital jammer" that works without knowing the actual audio content, testing whether Whisper has a structural vulnerability direction rather than only sample-specific weaknesses.

### 5.4 Targeted Carlini-Wagner (CW)

The targeted CW attack is implemented in `src/attacks/cw.py`. It minimizes a joint objective balancing perturbation size and target-phrase likelihood:

$$
\min_{\delta}\ \|\delta\|_2^2 + c \cdot L\!\left(f(x + \delta),\ y_{\text{target}}\right)
$$

The implementation includes three practical features:

1. **Binary search over $c$** — finds the minimal perturbation magnitude that achieves the target phrase
2. **Cosine annealing** — for the Adam optimizer learning rate to prevent late-stage oscillation
3. **Periodic transcription checks** — with early stopping when the target phrase is detected

| Parameter | Value |
|---|---|
| Target phrase | "hello world" |
| Learning rate | 0.005 |
| $c$ (loss weight) | 50.0 |
| Optimization steps | 100 |
| Binary search steps | 7 |
| Optimizer | Adam + cosine annealing |

This attack demonstrates the most security-critical scenario: injecting a chosen phrase rather than merely causing degradation.

### 5.5 Universal Carlini-Wagner (UCW)

The UCW attack is implemented in `src/attacks/ucw.py` and extends the per-sample CW framework into the universal setting. It trains a single shared perturbation $\delta$ that forces a target phrase on any audio input:

$$
\min_{\delta}\; \mathbb{E}_{x \sim X}\!\left[c \cdot L\!\left(f\!\left(x + \text{tile}(\delta)\right),\ y_{\text{target}}\right)\right] \quad \text{s.t.} \quad \|\delta\|_\infty \le \epsilon
$$

This is the most demanding optimization in the project. The perturbation must work across variable-length utterances from diverse speakers while producing a specific target phrase.

| Parameter | Value |
|---|---|
| Target phrase | "access evil website" |
| $\delta$ length | 160,000 samples (10.0 seconds) |
| $\epsilon$ (L$_\infty$ bound) | 0.02 |
| $c$ (CE weight) | 1.0 |
| Learning rate | $5 \times 10^{-4}$ |
| Epochs | 200 |
| Batch size | 1 |
| Gradient accumulation | 4 (effective batch = 4) |
| Initialization noise | $10^{-4}$ |
| Optimizer | PGD sign-gradient + cosine annealing |
| Early stopping | 20 epochs without validation improvement |

The training used PGD sign-gradient updates rather than Adam, because Adam's adaptive step sizes misalign with $L_\infty$ geometry. The L2 penalty term was removed after finding it consumed approximately 50% of the gradient budget and pushed $\delta \to 0$ without improving convergence. Cosine annealing decays the learning rate to 1% of the initial value to prevent late-stage oscillation.

Training used 1,600 files and 350 validation files from LibriSpeech test-clean across 40 speakers each. Training was halted by early stopping. The trained perturbation is saved at `results/ucw_delta.pt`.

**Key design evolution:**

| Change | Reason |
|---|---|
| L2 penalty removed | Consuming ~50% of gradient, pushing $\delta \to 0$ |
| Adam → PGD sign-gradient | Adam's adaptive steps misaligned with $L_\infty$ geometry |
| Cosine annealing added | Prevents late-stage oscillation |
| $\epsilon = 0.02$ (not 0.03) | Balances success rate and audibility |
| 10 s perturbation (not 5 s) | Better mel-spectrogram coverage for targeted injection |

---

## 6. Tools and Software Stack

The implementation uses a research stack captured in `requirements.txt`:

| Component | Purpose |
|---|---|
| PyTorch | Gradient computation and optimization |
| transformers (HuggingFace) | `WhisperForConditionalGeneration` model loading |
| openai-whisper | Baseline evaluation |
| librosa, soundfile | Audio loading and inspection |
| jiwer | WER and CER computation |
| matplotlib | Analysis plots |
| gradio, resampy | Interactive demo interface |

The core project code is organized under `src/`:

| Module | File | Purpose |
|---|---|---|
| Data loading | `src/data/audio_loader.py` | Load, resample, normalize audio; pad/crop to fixed lengths |
| Data download | `src/data/download_data.py` | Download LibriSpeech via torchaudio |
| Whisper wrapper | `src/models/whisper_wrapper.py` | Differentiable Whisper inference with gradient flow |
| PGD attack | `src/attacks/pgd.py` | Per-sample untargeted perturbation |
| UAP attack | `src/attacks/uap.py` | Universal untargeted perturbation |
| CW attack | `src/attacks/cw.py` | Per-sample targeted perturbation |
| UCW attack | `src/attacks/ucw.py` | Universal targeted perturbation |
| Base class | `src/attacks/base.py` | Shared apply/save/load for universal attacks |
| Utilities | `src/attacks/utils.py` | Text normalization, SNR, tile-to-length |

---

## 7. Training and Evaluation Protocol

### 7.1 Training Choices

The project follows practical constraints for adversarial optimization on ASR models:

- **16 kHz domain**: All perturbations are optimized directly at Whisper's expected sample rate to avoid resampling artifacts that would destroy the adversarial signal.
- **Waveform clamping**: Explicit `torch.clamp(audio + δ, -1, 1)` prevents out-of-range audio that causes crackling artifacts.
- **Sequential optimization**: Batch size of 1 for all attack loops to avoid GPU out-of-memory errors; gradient accumulation provides effective larger batches where needed.
- **Frozen weights**: Only the adversarial perturbation is optimized; Whisper's parameters are never modified.
- **Reproducibility**: `torch.manual_seed(42)` and `np.random.seed(42)` at the start of every experiment.
- **Gradient through preprocessing**: Audio tensor has `requires_grad=True` before the mel-spectrogram computation, and all preprocessing uses differentiable PyTorch operations.

### 7.2 Metrics

| Metric | Definition | Usage |
|---|---|---|
| **WER** | Word Error Rate — proportion of incorrectly transcribed words | Primary degradation metric |
| **CER** | Character Error Rate — captures partial word corruption | Preferred for adversarial evaluation |
| **SNR** | Signal-to-noise ratio between clean and perturbed audio | Imperceptibility measure |
| **Fooling rate** | Proportion of samples where adversarial output differs from clean | UAP convergence signal |
| **Success rate** | Proportion of samples containing the target phrase | Targeted attack evaluation |

SNR is computed as:

$$
\text{SNR}(x, x_{adv}) = 10 \log_{10}\!\left(\frac{\sum x^2}{\sum (x_{adv} - x)^2}\right)
$$

### 7.3 Evaluation Notes

The current notebooks do not use a single identical metric reference across all experiments:

- The clean baseline evaluates WER/CER against LibriSpeech ground-truth transcripts.
- PGD and UAP notebooks evaluate transcription drift between clean and adversarial Whisper outputs, plus fooling rate and SNR.
- CW and UCW notebooks report target success rate and WER against the target phrase.

Results should be compared within each experiment setup rather than directly across attack families.

---

## 8. Results

### 8.1 Clean Baseline

From `results/baseline_metrics.json` and `notebooks/02_performance_evaluation.ipynb`:

| Metric | Value |
|---|---:|
| Mean WER (vs ground truth) | 4.78% |
| Mean CER (vs ground truth) | 1.98% |
| Samples evaluated | 50 |

Most samples are transcribed perfectly (WER = 0 on approximately 40% of the test set). Errors are primarily word-level (minor substitutions), with CER remaining very low.

### 8.2 Untargeted PGD Results

From `notebooks/03_pgd_attack.ipynb`:

| Metric | Value |
|---|---:|
| Attack parameters | $\epsilon = 0.01$, $\alpha = 0.001$, 20 iterations |
| Samples evaluated | 50 |
| **Attack success rate** | **92.0%** (46/50) |
| **Average WER drift** | **46.25%** |
| **Average SNR** | **26.58 dB** |
| SNR range | 20.71–40.62 dB |
| Average perturbation magnitude | 0.0099 |

**Interpretation:**

- PGD is highly effective at changing Whisper output on a per-sample basis, with 92% of utterances producing measurably different transcriptions.
- Mean WER increased from 4.78% (baseline) to 46.25% (attacked), representing a 10x degradation in transcription quality.
- SNR of 26.58 dB is below the ideal imperceptibility target of 35–45 dB but represents moderately perceptible perturbations.
- All perturbations respect the $\epsilon = 0.01$ $L_\infty$ constraint, confirming the projection step works correctly.

### 8.3 Universal Perturbation (UAP) Results

From `notebooks/04_uap_training.ipynb`:

| Metric | Value |
|---|---:|
| Training set | ~100 samples, 5 s each, $\epsilon = 0.08$ |
| Training epochs | 20 |
| UAP vector length | 80,000 samples (5.0 seconds) |
| Validation set | 20 held-out samples |
| **Fooling rate** | **60.0%** |
| Average WER drift | 35–45% |
| Perturbation saved | `results/universal_perturbation_v.pt` |

**Interpretation:**

- A single fixed perturbation can degrade 60% of unseen utterances without per-sample re-optimization.
- The UAP generalizes across different speakers and acoustic conditions despite training on only 10% of the dataset.
- CER exceeding 100% on some samples is possible with jiwer when the adversarial output contains many spurious insertions relative to a short reference string.
- The saved perturbation waveform shows structured noise (not Gaussian), indicating it has learned specific frequency patterns that exploit Whisper's mel-spectrogram processing.

The UAP result is especially important from an operational perspective because a universal perturbation could be applied repeatedly as a "digital jammer" without re-optimizing for every new utterance.

### 8.4 Targeted CW Results

From `notebooks/05_targeted_attack.ipynb`, using the target phrase **"hello world"**:

| Metric | Value |
|---|---:|
| Samples evaluated | 5 |
| Attack parameters | $c = 50.0$, 100 steps, Adam + cosine annealing |
| **Mean SNR** | **14.50 dB** |
| Mean WER (clean audio) | 5% |
| Mean WER (adversarial vs ground truth) | 55% |

**Interpretation:**

- Per-sample CW confirmed that targeted phrase injection is feasible in the white-box setting. The optimization finds perturbations that cause Whisper's decoder to produce the exact target phrase.
- Binary search over $c$ identifies the minimal perturbation magnitude needed, and cosine annealing learning rate prevents oscillation near convergence.
- The SNR of 14.50 dB indicates the perturbation is more audible than PGD, reflecting the harder optimization objective of controlling the exact output rather than merely degrading it.

### 8.5 Universal Targeted (UCW) Results

From `notebooks/06_universal_targeted_attack.ipynb`, using the target phrase **"access evil website"**:

**Training progression:**

| Epoch range | Training success rate |
|---|---|
| 1–50 | 0% → ~50% |
| 50–100 | ~50% → ~75–80% |
| 100–200 | Stabilizes at ~90% |

**Validation results (350 held-out samples):**

| Metric | Value |
|---|---:|
| **Success rate (contains target)** | **58.6%** (205/350) |
| **Success rate (exact match)** | **56.0%** (196/350) |
| Mean WER vs target phrase | 2.446 |
| Mean CER vs target phrase | 2.202 |
| **Mean SNR** | **11.34 dB** |
| $\delta$ min / max | $-0.02$ / $+0.02$ (at $\epsilon$ boundary) |
| Perturbation saved | `results/ucw_delta.pt` |

**Interpretation:**

- **Universal targeted attacks on Whisper are feasible.** A single fixed perturbation forces the target phrase "access evil website" on 58.6% of 350 unseen utterances across 40 diverse speakers.
- The **training-to-validation gap** (~90% train vs. 58.6% val) indicates partial overfitting to the training distribution. The perturbation learns directions that generalize broadly but not perfectly.
- Mean WER vs target (2.446) and CER vs target (2.202) reflect that most adversarial outputs either exactly match the target phrase or produce completely degraded text, with few partial matches.
- At 11.34 dB SNR, the perturbation is more audible than the per-sample attacks, reflecting the additional difficulty of universal optimization.
- The perturbation saturates the $\epsilon$ boundary ($\pm 0.02$), indicating higher $\epsilon$ could improve success at the cost of further reduced SNR.

### 8.6 Cross-Attack Comparison

| Property | PGD | UAP | CW | UCW |
|---|---|---|---|---|
| **Type** | Per-sample untargeted | Universal untargeted | Per-sample targeted | Universal targeted |
| **Samples** | 50 | 20 (val) | 5 | 350 (val) |
| **$\epsilon$ / $L_\infty$** | 0.01 | 0.08 | ~0.03–0.05 (L2) | 0.02 |
| **Success rate** | **92%** | 60% | Demonstrated | **58.6%** |
| **Mean SNR** | **26.58 dB** | ~16–18 dB | **14.50 dB** | **11.34 dB** |
| **Deploy speed** | Slow (per-sample) | Fast (O(1)) | Slow (per-sample) | **Fast (O(1))** |
| **Security impact** | Degradation | Digital jammer | Phrase injection | **Scalable injection** |

The clear tradeoffs: per-sample attacks achieve higher success rates and better SNR, but require computational access to the model at attack time. Universal attacks sacrifice success rate and SNR for deployability — a single pre-computed perturbation can be applied in real time without re-optimization.

---

## 9. Interactive Demo

The project includes a full interactive Gradio demo implemented in `notebooks/08_train_demo_targeted_attack.ipynb`. The demo loads pre-trained perturbations and supports two interfaces:

### 9.1 Single-Audio Demo

Users can upload or record audio, select an attack mode (Clean, Untargeted UAP, or Targeted CW Injection), and compare:

- Clean transcript vs. adversarial transcript
- Signal-to-noise ratio
- Waveform comparison plot
- Playback of the adversarial audio

### 9.2 Live Streaming Demo

A real-time interface that continuously transcribes microphone audio with attack perturbations applied:

| Feature | Detail |
|---|---|
| Live preview | Re-transcribes the current utterance every ~1 second |
| History logging | Commits finalized utterances on silence boundaries |
| Overlap handling | Removes repeated boundary words in history |
| Attack modes | Same three modes: Clean, UAP, CW Injection |

The streaming implementation includes adaptive noise-floor estimation, frame-level VAD (voice activity detection), and mode-specific thresholds. For the targeted CW mode, the streaming system accumulates longer audio segments (minimum 3 seconds of speech) before committing, because the 10-second perturbation requires sufficient audio length for the full adversarial pattern to be effective in the mel spectrogram. Short live previews skip the CW perturbation and show clean transcripts instead, as fragments under 3 seconds truncate the perturbation and produce unreliable results.

The audio is zero-padded to the perturbation length (160,000 samples) before applying the targeted CW delta, ensuring the full adversarial pattern is present regardless of segment duration. This padding is harmless because Whisper internally pads all audio to 30 seconds.

---

## 10. Discussion

### 10.1 Central Finding

Whisper's robustness to natural noise does not carry over to adversarial noise (Olivier & Raj, 2023). Despite being trained on 680,000 hours of diverse audio, Whisper-base is vulnerable to all four attack families tested. The perturbations exploit the model's mel-spectrogram decision boundary through gradient-based optimization in the waveform domain.

### 10.2 Operational Significance

For this attack to be effective in production scenarios, affecting a transcription corpus is more important than achieving 100% effectiveness. As transcriptions are poisoned by the perturbation, it reduces the credibility of all existing transcriptions, jeopardizing trust in the tool.

The most significant combined finding is:

- A **reusable untargeted perturbation** (UAP) that disrupts 60% of utterances represents a real-world digital jammer.
- A **universal targeted perturbation** (UCW) that injects a specific phrase on more than half of diverse utterances represents a scalable prompt-injection vector at the audio layer.

### 10.3 Imperceptibility Tradeoff

The most successful attacks do not yet operate in the ideal 35–45 dB SNR range associated with imperceptible perturbations:

| Attack | Mean SNR | Imperceptibility |
|---|---:|---|
| PGD | 26.58 dB | Moderately perceptible |
| UAP | ~16–18 dB | Noticeable |
| CW | 14.50 dB | Clearly audible |
| UCW | 11.34 dB | Most audible |

This limits immediate real-world stealth, especially for targeted attacks. The current performance implies a real vulnerability in digital pipelines, batch transcription settings, or controlled scenarios, but not yet an ideal covert attack under realistic human listening conditions.

### 10.4 Limitations

- Audio files are not speaker-stratified across all experiments, which may introduce biased fooling-rate estimates.
- Evaluation metrics are not fully standardized across attack families, limiting direct cross-attack comparison.
- All results are in the digital domain; over-the-air physical playback would introduce additional degradation.
- The UCW training-to-validation gap (90% to 58.6%) suggests room for improved generalization.

---

## 11. Future Work

Three main directions for improvement:

### Attack Strength and Imperceptibility

The primary target is improving the tradeoff between effectiveness and SNR. Perceptually informed loss functions (e.g., incorporating psychoacoustic masking models) could maintain attack success while keeping perturbations less audible, especially for targeted and universal attacks.

### Standardized Evaluation

Future experiments should use one consistent protocol for text normalization, ground-truth comparison, success criteria, and perceptual quality metrics so PGD, UAP, CW, and UCW results can be compared directly. The pipeline should also be tested beyond LibriSpeech test-clean, including cross-dataset and cross-language settings.

### Transferability and Realism

The strongest research opportunities include:

- **Over-the-air attacks** — testing whether perturbations survive physical playback and recording
- **Black-box transfer** — evaluating whether perturbations transfer to wav2vec 2.0 or other Transformer-based ASR
- **Multilingual robustness** — extending attacks beyond English
- **Improved universal targeting** — curriculum training, larger and more diverse training sets, warm-starting from untargeted UAP

---

## 12. Conclusion

This capstone demonstrates that Whisper-base can be attacked in four distinct ways: by degrading a single transcription (PGD, 92% success), by learning a reusable universal jammer (UAP, 60% fooling rate), by forcing a targeted phrase per sample (CW), and by universally injecting a phrase across diverse utterances (UCW, 58.6% on 350 held-out samples).

Although targeted attacks show lower success rates than untargeted ones, their implications for agentic pipelines are significant. As transcription systems are increasingly integrated into LLM inference and execution infrastructure, even a single successful targeted injection transferred to the downstream pipeline cannot be ignored.

The combination of UAP and UCW findings is the most significant result. A reusable untargeted perturbation that disrupts the majority of utterances, combined with a universal targeted perturbation that injects a specific phrase on more than half of diverse utterances, represents a real and scalable threat vector at the audio layer.

The project also surfaces the central open engineering question: how to push attack success rates higher while keeping SNR in a genuinely imperceptible range. The SNR limitations (26.58 dB for PGD, 14.50 dB for CW, 11.34 dB for UCW) show that current attacks trade audibility for effectiveness. Improving imperceptibility while maintaining attack success is the primary challenge for future work.

As ASR systems become the transaction layer between human speech and automated decision-making infrastructure, the vulnerabilities documented here represent a category of attack that should be considered by any team deploying Whisper or similar models in security-sensitive, real-time, or agentic contexts.

---

## References

Airlangga, G. (2023). Evaluating the efficacy of traditional machine learning models in speaker recognition: A comparative study using the LibriSpeech dataset. *Brilliance: Research of Artificial Intelligence*, 3(2), 90–101. https://doi.org/10.47709/brilliance.v3i2.3488

Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks (arXiv:1608.04644). https://doi.org/10.48550/arXiv.1608.04644

Carlini, N., & Wagner, D. (2018). Audio adversarial examples: Targeted attacks on speech-to-text (arXiv:1801.01944). https://doi.org/10.48550/arXiv.1801.01944

Chen, Y., Chen, H., Qiao, Y., Yun, X., & Zhao, Z. (2023). Near-ultrasound inaudible Trojan (Nuit): Exploiting your speaker to attack your microphone. In *Proceedings of the 32nd USENIX Security Symposium*. USENIX Association. https://www.usenix.org/node/287267

Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2019). Towards deep learning models resistant to adversarial attacks (arXiv:1706.06083). https://doi.org/10.48550/arXiv.1706.06083

Moosavi-Dezfooli, S.-M., Fawzi, A., Fawzi, O., & Frossard, P. (2017). Universal adversarial perturbations. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

Neekhara, P., Hussain, S., Pandey, P., Dubnov, S., McAuley, J., & Koushanfar, F. (2019). Universal adversarial perturbations for speech recognition systems. *Interspeech 2019*.

Olivier, R., & Raj, B. (2022). There is more than one kind of robustness: Fooling Whisper with adversarial examples (arXiv:2210.17316). https://arxiv.org/abs/2210.17316

Olivier, R., & Raj, B. (2023). Fooling Whisper with adversarial examples. *Interspeech 2023*. https://www.isca-archive.org/interspeech_2023/olivier23_interspeech.pdf

OpenAI. (2022, September 21). *Introducing Whisper*. https://openai.com/index/whisper/

Pratap, V., Xu, Q., Sriram, A., Synnaeve, G., & Collobert, R. (2020). MLS: A large-scale multilingual dataset for speech research. arXiv. https://arxiv.org/abs/2012.03411

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision (arXiv:2212.04356). https://doi.org/10.48550/arXiv.2212.04356

TensorFlow. (2024, December 10). *librispeech | TensorFlow Datasets*. https://www.tensorflow.org/datasets/catalog/librispeech

Yuan, X., Chen, Y., Zhao, Y., Long, Y., Liu, X., Chen, K., Zhang, S., Huang, H., Wang, X., & Gunter, C. A. (2018). CommanderSong: A systematic approach for practical adversarial voice recognition (arXiv:1801.08535). https://arxiv.org/abs/1801.08535

Zhang, G., Yan, C., Ji, X., Zhang, T., Zhang, T., & Xu, W. (2017). DolphinAttack: Inaudible voice commands (arXiv:1708.09537). https://arxiv.org/abs/1708.09537



