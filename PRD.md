***

## Audio Adversarial Attacks on ASR (Whisper): A 6-Week Capstone Project Plan
### Executive Summary
This document provides a comprehensive technical blueprint for implementing adversarial attacks on OpenAI's Whisper speech recognition model. The project focuses on generating imperceptible audio perturbations that cause transcription failures—a critical security vulnerability with implications for content moderation, command injection, and data poisoning. The **universal perturbation approach** is recommended as the most efficient methodology for a 6-week timeline, balancing technical novelty, implementation feasibility, and research rigor.

***

### Part 1: How Adversarial Audio Attacks on ASR Work
#### **1.1 Fundamental Concept: The Adversarial Perturbation**

Adversarial attacks on speech recognition exploit a critical vulnerability: while Whisper is robust to natural noise and random perturbations, it is highly vulnerable to **adversarial noise**—small, crafted modifications specifically optimized to fool the model.[1]

The attack adds a small perturbation **δ** to a clean audio signal **x** such that:
- **x_adversarial = x + δ**
- **||δ||_∞ < ε** (the perturbation magnitude is bounded)
- **Human listeners cannot perceive the change** (imperceptible noise, 30-45 dB Signal-to-Noise Ratio)
- **Whisper's transcription dramatically fails** (Word Error Rate degrades 35-99%)

The key insight: these perturbations exploit geometric vulnerabilities in Whisper's decision boundaries—not flaws in microphone hardware or audio compression.
#### **1.2 The Three Attack Methodologies**
**Attack Type 1: Untargeted PGD (Projected Gradient Descent)**

The simplest attack. The adversary maximizes the transcription loss without caring what garbage the model outputs.

*Mathematical Goal*:
```
max ||δ||_p < ε  L(f(x + δ), y)
```
Where:
- **L** = Cross-entropy loss function
- **f** = Whisper model
- **y** = Original correct transcription
- **ε** = Maximum perturbation magnitude

*Why it works*: 
- The gradient ∇_δ L tells you exactly how to push the audio to cross the decision boundary
- You take small steps in the direction of steepest loss increase
- Every step is projected back into the ε-ball to stay imperceptible

*Results on Whisper*:[1]
- **40dB SNR**: 35-89% absolute WER degradation (model output becomes unintelligible)
- **Execution time**: ~2 minutes per utterance (A100 GPU)
- **All Whisper sizes vulnerable**: From 39M to 1550M parameters

**Attack Type 2: Targeted CW (Carlini-Wagner)**

The attacker forces Whisper to output a specific phrase (e.g., "OK Google, browse to evil.com").

*Mathematical Goal*:
```
min ||δ||_∞ < ε  [L(f(x + δ), y_target) + c·||δ||_2²]
```

*Why it's harder*:
- You must simultaneously: (1) Match a specific target transcription, (2) Keep perturbation small, (3) Navigate sequence-to-sequence loss landscape
- The first predicted token is particularly resistant—requires special loss weighting: `λ = 1` on first token[1]

*Results*:
- **Success rate**: 50-90% (depends on model size and target phrase)
- **SNR**: 35-45dB (imperceptible)
- **Execution time**: ~25 minutes per utterance (2000 iterations)
- **Multilingual models are harder to fool** (+5-10dB SNR penalty), suggesting language diversity provides some defense

**Attack Type 3: Universal Adversarial Perturbations (UAP) — THE BEST FOR CAPSTONE**

Instead of computing a unique perturbation for each audio file, compute **one single perturbation** that fools the model on *any* input.

*The Game Changer*: Once computed, the perturbation is pre-calculated and reusable.

*Algorithm *:[2]
```
Initialize v ← 0 (zero perturbation)

While success_rate(test_set) < target_success_rate:
    For each training audio xi:
        Compute minimum additional perturbation Δvi that makes:
            CER(Whisper(xi + v + Δvi), correct_transcription) > 0.5
        
        Update universal perturbation:
            v ← Clip(v + Δvi, ε)  // Keep in norm-ball
            
        Evaluate on validation set
```

*Why Universal Perturbations Win for Your Capstone*:

1. **No per-input optimization**: Compute once, apply forever
2. **Real-time attack potential**: No latency—just add pre-computed noise in real-time
3. **Cross-model transferability**: Perturbation trained on DeepSpeech achieves 63% success on WaveNet[2]
4. **Massive data volume**: Attack 1000s of utterances in one batch experiment
5. **Research novelty**: Enables multi-modal studies (languages, domains, model architectures)

*Results *:[2]
- **Success Rate**: 89% on DeepSpeech validation set
- **Mean distortion**: -32 dB (roughly the acoustic difference between a quiet room and normal speech)
- **Training data required**: Only 1,000 examples achieve 80% success rate
- **Generalization**: Works on architecturally different models (42%+ success on unseen WaveNet)

***

### Part 2: Project Implementation Plan
#### **2.1 Technical Architecture**

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Clean Audio (LibriSpeech, CommonVoice)     │
│  Format: PCM 16kHz wav                              │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  ATTACK GENERATION                                   │
│  ┌─────────────────────────────────────────────┐   │
│  │ CTC Loss + Gradient Descent Optimization    │   │
│  │ (SpeechBrain + PyTorch autograd)           │   │
│  └─────────────────────────────────────────────┘   │
│  Output: Perturbation δ (shape: [16000*duration])  │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  PERTURBATION APPLICATION                           │
│  x_adv = clip(x + δ, audio_range)                  │
│  Ensure: SNR(δ, x) ∈ [30-45] dB (imperceptible)   │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  WHISPER ASR PROCESSING                             │
│  ┌─────────────────────────────────────────────┐   │
│  │ Audio → Mel spectrogram → Transformer Enc   │   │
│  │ → Sequence-to-sequence decoder → Text       │   │
│  └─────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  EVALUATION METRICS                                  │
│  • Word Error Rate (WER)                            │
│  • Character Error Rate (CER)                       │
│  • Success Rate (% of utterances with CER > 0.5)   │
│  • SNR / Imperceptibility                           │
└──────────────────────────────────────────────────────┘
```

#### **2.2 Software Stack**

| Component | Tool | Reason |
| --- | --- | --- |
| **Model** | OpenAI Whisper (open-source) | White-box access; 5 model sizes; multilingual |
| **Optimization** | PyTorch + autograd | Gradient computation through MFCC + model |
| **Framework** | SpeechBrain | Pre-built ASR pipeline; CTC loss implementation |
| **Defense eval** | Robust Speech library | WaveGuard defense mechanism |
| **Data** | LibriSpeech + CommonVoice | 75-100 utterances per experiment |
| **Compute** | NVIDIA GPU (16GB+ VRAM) | Critical: A100 recommended for CW attacks (~25 min/utterance) |

#### **2.3 Six-Week Timeline**
**Week 1: Foundation & Setup**
- Day 1-2: Literature review (read Olivier 2023, Neekhara 2019 papers in full)
- Day 3-4: Environment setup
  - Clone Whisper repo: `pip install openai-whisper`
  - Install SpeechBrain: `pip install speechbrain`
  - Verify GPU access (CUDA + cuDNN)
- Day 5-7: Dataset preparation
  - Download LibriSpeech test-clean (~10 GB, 75 utterances)
  - Download CommonVoice sample (10-20 multilingual utterances)
  - Write preprocessing script: normalize audio to 16 kHz, verify SNR calculation

**Week 2-3: Untargeted PGD Implementation**
- Day 8-10: Implement basic PGD attack
  ```python
  # Pseudocode structure
  for iteration in range(200):
      loss = whisper_loss(audio + delta, original_transcript)
      grad = compute_gradient(loss, delta)
      delta = delta + learning_rate * grad
      delta = clip(delta, epsilon)  # Project to Lp-ball
  ```
- Day 11-14: Experimentation
  - Test L2 vs L∞ norm constraints
  - Optimize hyperparameters (learning rate, epsilon, iterations)
  - Measure WER degradation on 10 utterances
  - Document SNR of generated perturbations
- Day 15: Analysis & reporting
  - Create visualization: WER vs SNR tradeoff
  - Compare results to Olivier et al. baselines

**Week 4: Targeted CW Implementation (Optional/Bonus)**
- Day 22-24: Implement modified CW attack with first-token loss weighting
  - Risk: Takes significant time (~25 min per utterance)
  - Recommendation: If ahead of schedule, pursue; otherwise skip for universal perturbations
- Day 25-28: Targeted attack evaluation on 5 utterances

**Week 4-5: Universal Perturbation Algorithm (Core)**
- Day 22-25: Implement universal perturbation training loop
  ```python
  v = zeros(sample_rate * duration)  # 30-second perturbation
  
  for epoch in range(2000):
      for audio_xi in training_set:
          # Compute min perturbation to mis-transcribe
          delta_i = minimize(
              ctc_loss(whisper(audio_xi + v + delta_i), correct_transcript),
              constraint=||delta_i||_2 < epsilon
          )
          v = clip(v + delta_i, epsilon)
      
      # Check success rate on validation set
      success_rate = evaluate(v, validation_set)
      print(f"Epoch {epoch}: Success rate = {success_rate}%")
  ```
- Day 26-28: Training & hyperparameter tuning
  - Training set: 70 utterances; validation: 20 utterances; test: 75 utterances
  - Hyperparameters: learning_rate=0.001*ε, regularization_c=0.5, SNR_target=40dB
  - Monitor: Success rate should reach 85%+ by epoch 500

**Week 5-6: Evaluation & Defense**
- Day 29-32: Comprehensive evaluation on test set
  - Primary metric: Success rate (%), Mean CER, Mean SNR
  - Analysis: Does universal perturbation transfer to Whisper-large? Other languages?
  - Comparison to random noise baseline
- Day 33-35: Test defense mechanisms
  - Implement randomized smoothing defense (add Gaussian noise σ=0.02, 0.03)
  - Measure trade-off: WER increase on clean audio vs. attack success rate reduction
  - Optional: Adversarial training (include adversarial examples in a fine-tuning set)
- Day 36-42: Report writing & visualization
  - Results figures: WER degradation curves, success rate vs. SNR
  - Technical writeup: algorithm pseudocode, theoretical justification
  - Discussion: practical security implications, limitations

***

### Part 3: Specific Technical Details
#### **3.1 Loss Functions & Gradients**

**CTC (Connectionist Temporal Classification) Loss**

Whisper uses CTC loss during training (modified for decoding). CTC handles alignment between variable-length audio and variable-length transcriptions:

```
L_CTC = -log P(y | x)
```

Where `y` is the target transcription and `P(y | x)` is the likelihood of the target given input audio. During adversarial optimization, you backpropagate this loss through the entire Mel-spectrogram computation and neural network.

**Key insight for CW attack**: Standard CTC loss makes first tokens hard to target. Solution: Use weighted loss:

```
L_weighted = (1 + λ) * L_CTC(first_token) + Σ L_CTC(remaining_tokens)
```

With λ=1, the first token carries 2x weight.[1]

#### **3.2 SNR Calculation (Imperceptibility Metric)**

Signal-to-Noise Ratio in dB is the standard metric for imperceptibility:

```
SNR(δ, x) = 20 * (log10(||x||_2) - log10(||δ||_2))
```

**Reference scale**:[2]
- **50 dB**: Almost completely imperceptible (background whisper-level noise)
- **40 dB**: Imperceptible to most humans (subtle hiss)
- **30 dB**: Just perceptible (noticeable but not annoying)
- **20 dB**: Obvious noise (degraded audio quality)

For your project:
- **Target SNR**: 35-45 dB (imperceptible to humans, but sufficient to fool Whisper)
- **Measure SNR** for all generated perturbations and report in results

#### **3.3 Evaluation Metrics**

**Word Error Rate (WER)**:
```
WER = (S + D + I) / N × 100%
```
Where S = substitutions, D = deletions, I = insertions, N = total words in reference

**Character Error Rate (CER)**:
```
CER = (S + D + I) / N × 100%
```
Same as WER but character-level (more sensitive)

**Success Rate**:
```
Success_Rate = (# utterances with CER > 50%) / (total utterances) × 100%
```

Why CER > 50%? Because a single character change is not sufficient evidence of attack success. At least half the output must be corrupted.[2]

***

### Part 4: Research Novelty Angles
To elevate your capstone beyond "implemented known attack," consider these research questions:

#### **Angle 1: Cross-Language Universal Perturbations**
*Question*: Does a universal perturbation trained on English audio transfer to other languages?

*Experiment*:
- Train perturbation on 70 English utterances from LibriSpeech
- Test on 20 utterances from CommonVoice (Arabic, Mandarin, French, etc.)
- Hypothesis: Language-agnostic perturbations should transfer moderately (exploit low-level acoustic features)
- Novelty: First empirical study of universal UAP cross-language transfer on modern ASR

#### **Angle 2: Defense-Aware Adversarial Examples**
*Question*: Can you generate perturbations that evade randomized smoothing defense?

*Experiment*:
- Assume attacker knows defender will use Gaussian smoothing (σ=0.02)
- Incorporate defense into adversarial training loop: `loss = L_CTC(whisper(audio + delta + gaussian_noise), target)`
- Compare defense-aware vs. defense-unaware perturbations
- Novelty: Adaptive adversarial attack design for known defenses

#### **Angle 3: Minimal Perturbation Bounds**
*Question*: What is the theoretical minimum SNR at which universal perturbations can achieve X% success?

*Experiment*:
- Vary allowed perturbation magnitude (||v||_∞) from 0.001 to 0.01
- For each magnitude, train universal perturbation and measure success rate
- Plot: success rate vs. SNR (create a tradeoff curve)
- Compare to random noise baseline
- Novelty: Quantify the vulnerability surface of Whisper (how much robustness is missing?)

#### **Angle 4: Phonetic Analysis of Adversarial Perturbations**
*Question*: What phonetic/acoustic features do universal perturbations target?

*Experiment*:
- Spectral analysis: Compute Mel-spectrogram of universal perturbation
  - Are there frequency bands that matter most?
  - Does perturbation target formants of vulnerable phonemes?
- Speaker/accent sensitivity: Does the same perturbation work across accents?
- Novelty: Provide interpretability to the "black box" attack

***

### Part 5: Potential Challenges & Mitigations
| Challenge | Mitigation |
| --- | --- |
| **GPU memory limits** | Start with Whisper-base (74M params); reduce batch size if OOM |
| **Slow CW attack** | Skip CW in Week 4; focus on universal perturbations (more valuable) |
| **Library compatibility** | Test all imports in Week 1; have backup: manual Whisper implementation |
| **Dataset download latency** | Start LibriSpeech download in Week 1, use smaller CommonVoice subset |
| **Non-convergent optimization** | Monitor loss curves; increase learning rate or iteration count |
| **Evaluation variance** | Use 3 random seeds; report mean ± std dev |

***

### Part 6: Deliverables & Success Criteria
**For a Master's-level capstone, your report should demonstrate**:

1. **Reproducible Implementation**
   - Clean, commented code (GitHub repo)
   - Dockerfile/requirements.txt for reproducibility
   - Ability to regenerate all results

2. **Comprehensive Evaluation**
   - Minimum 50 test utterances; report success rate ± confidence interval
   - SNR measurements for all perturbations
   - Comparison to random noise baseline
   - At least one novel evaluation angle (cross-language, cross-model, etc.)

3. **Theoretical Understanding**
   - Explain CTC loss and gradient backpropagation
   - Justify why universal perturbations are more efficient than per-sample attacks
   - Discuss defense mechanisms and their trade-offs

4. **Limitations & Ethical Discussion**
   - Acknowledge practical limitations (white-box access, digital domain)
   - Discuss real-world deployment barriers (physical room acoustics)
   - Address security/ethics: who should hear about these vulnerabilities, when, and how?

***
Olivier, R., & Raj, B. (2023). "Fooling Whisper with adversarial examples." *Interspeech 2023*.[1]

 Neekhara, P., Hussain, S., Pandey, P., Dubnov, S., McAuley, J., & Koushanfar, F. (2019). "Universal adversarial perturbations for speech recognition systems." *Interspeech 2019*.[2]

Additional key papers:
- Carlini, N., & Wagner, D. (2018). "Audio adversarial examples: Targeted attacks on speech-to-text." IEEE S&P workshops.
- Hussain, S., et al. (2021). "Understanding and mitigating audio adversarial examples." *USENIX Security*.

***

This plan gives you everything needed for a rigorous, publishable 6-week capstone project. The **universal perturbation approach** offers the optimal balance of feasibility, novelty, and impact. Good luck!

[1](https://www.isca-archive.org/interspeech_2023/olivier23_interspeech.pdf)
[2](https://cseweb.ucsd.edu/~jmcauley/pdfs/interspeech19b.pdf)
[3](https://arxiv.org/html/2501.11378v1)
[4](https://www.usenix.org/system/files/sec21fall-hussain.pdf)
[5](https://web.eecs.utk.edu/~jliu/publications/xie2021enabling.pdf)
[6](https://divis.io/en/2024/04/whisper3-large-java-djl/)
[7](https://www.jetir.org/papers/JETIR2511538.pdf)
[8](https://www.isca-archive.org/interspeech_2019/neekhara19b_interspeech.pdf)
[9](https://github.com/rainavyas/prepend_acoustic_attack)
[10](https://arxiv.org/html/2411.09220v1)
[11](https://mosis.eecs.utk.edu/publications/xie2021realtime.pdf)
[12](https://github.com/jiakaiwangCN/awesome-physical-adversarial-examples)
[13](https://dl.acm.org/doi/10.1145/3716553.3750779)
[14](https://arxiv.org/abs/1905.03828v2)
[15](https://github.com/Trustworthy-AI-Group/Adversarial_Examples_Papers)
[16](https://ieeexplore.ieee.org/iel8/6287639/10820123/11175377.pdf)
[17](https://arxiv.org/abs/1905.03828)
[18](https://github.com/openai/whisper)
[19](https://github.com/hammaad2002/ASRAdversarialAttacks)
[20](https://www.isca-archive.org/interspeech_2019/neekhara19b_interspeech.html)



# Implementation

You can structure this capstone as a small but clean research codebase with clear separation between data, models, attacks, and evaluation, and your first steps should focus on environment setup, a minimal Whisper + dataset pipeline, and then a basic untargeted PGD loop before moving to universal perturbations.[1]

## Recommended repo structure

Use a research-style layout so you can iterate on attacks without breaking the core ASR pipeline.[1]

```text
audio-adversarial-whisper/
│
├─ README.md               # How to run experiments, reproduce results
├─ requirements.txt        # whisper, speechbrain, torch, numpy, librosa, jiwer, etc.
├─ configs/
│   ├─ data.yaml           # Paths, sample rates, splits
│   ├─ attack_pgd.yaml     # ε, steps, lr, norms
│   ├─ attack_uap.yaml     # epochs, ε, SNR targets
│   └─ eval.yaml           # metrics and evaluation settings
│
├─ data/
│   ├─ raw/
│   │   ├─ librispeech/
│   │   └─ commonvoice/
│   └─ processed/
│       ├─ wav16k/         # Resampled, normalized
│       └─ metadata.csv    # file_path, transcript, split
│
├─ src/
│   ├─ __init__.py
│   ├─ datasets.py         # LibriSpeech/CommonVoice loaders, resampling, batching
│   ├─ whisper_wrapper.py  # Clean API for forward pass, loss, decoding
│   ├─ metrics.py          # WER, CER, SNR utilities
│   ├─ attacks/
│   │   ├─ __init__.py
│   │   ├─ pgd.py          # Per-utterance untargeted PGD attack
│   │   └─ uap.py          # Universal perturbation training loop
│   ├─ defenses/
│   │   └─ smoothing.py    # Gaussian noise, etc. (later in project)
│   ├─ utils/
│   │   ├─ audio_io.py     # Load/save wav, clipping, normalization
│   │   └─ logging.py      # Experiment logging, seeding
│   ├─ train_uap.py        # Script to train universal perturbation
│   ├─ run_pgd_experiment.py
│   └─ evaluate.py         # Run WER/CER/SNR on clean vs adversarial
│
└─ notebooks/
    ├─ 01_explore_dataset.ipynb
    ├─ 02_whisper_baseline.ipynb
    └─ 03_pgd_sanity_check.ipynb
```

This aligns with the plan’s emphasis on separating attack generation, perturbation application, and evaluation (WER/CER/SNR) for clarity and reproducibility.[1]

## Step 1: Environment and baseline pipeline

The goal here is: one command that runs Whisper on a small test set and prints WER/CER, before any attack.[1]

- Set up environment:
  - Create a virtualenv/conda env with Python 3.10+.
  - Install core dependencies:
    - `pip install openai-whisper speechbrain torch torchvision torchaudio jiwer librosa` (pin versions in `requirements.txt`).[1]
- Implement `whisper_wrapper.py`:
  - Load a chosen model (start with `base` or `small` for speed).[1]
  - Provide functions:
    - `transcribe(audio_tensor) -> text`
    - `loss(audio_tensor, transcript) -> scalar` using CTC or sequence loss as in the document.[1]
- Prepare data:
  - Download LibriSpeech `test-clean` subset and a small CommonVoice sample (10–20 multilingual utterances).[1]
  - Write a script in `datasets.py` that:
    - Resamples all audio to 16 kHz.
    - Normalizes amplitudes and stores paths and transcripts in a `metadata.csv`.[1]
- Implement metrics:
  - In `metrics.py`, add:
    - WER and CER using the standard formulas with substitutions, insertions, and deletions.[1]
    - SNR computation \( \text{SNR}(\delta, x) = 20 (\log_{10} \|x\|_2 - \log_{10} \|\delta\|_2) \). [1]
- Baseline check:
  - Create `evaluate.py` to:
    - Run Whisper on 20–50 clean utterances.
    - Report baseline WER and CER so you know the attack’s effect later.[1]

## Step 2: Implement untargeted PGD attack

Next, implement a simple per-sample untargeted PGD attack to validate the optimization loop and gradients.[1]

- Design the PGD function in `attacks/pgd.py`:
  - Inputs: clean audio \(x\), correct transcript \(y\), model, loss_fn, ε, steps, step_size, norm \(L_\infty\) or \(L_2\).[1]
  - Loop:
    - Initialize `delta = 0` with `requires_grad=True`.
    - For `T` iterations:
      - Compute `loss = loss_fn(x + delta, y)` (maximize this loss).[1]
      - Backprop to get `∇_δ loss`.
      - Gradient ascent step on `delta`.
      - Project back into the ε-ball and clip audio to valid range.
- Add a script `run_pgd_experiment.py`:
  - Run PGD on a small subset (10–20 utterances).
  - Measure:
    - WER/CER before and after attack.
    - SNR of perturbations to keep them in the 30–45 dB range for imperceptibility.[1]
- Sanity checks:
  - Ensure:
    - Human-audible difference is small when listening.
    - WER degradation is significant (goal: large increase in WER/CER vs baseline).[1]

## Step 3: Set up universal perturbation training loop

Once PGD works, build the universal perturbation training infrastructure, even if you do not fully train it yet.[1]

- Implement `attacks/uap.py`:
  - Maintain a global perturbation `v` initialized to zeros with length corresponding to your maximum clip duration (e.g., 30 s).[1]
  - For each epoch:
    - For each training utterance \(x_i\):
      - Optimize a small `Δv_i` (similar to PGD) that, when added to `v`, makes CER exceed a threshold (e.g., > 50%).[1]
      - Update `v ← clip(v + Δv_i, ε)` to stay within the allowed norm.[1]
    - Evaluate current `v` on a validation set to compute success rate.[1]
- Implement `train_uap.py`:
  - Define train/validation/test splits (e.g., 70/20/75 utterances as suggested).[1]
  - Log:
    - Success rate (percentage of utterances with CER > 0.5).[1]
    - Mean SNR of adversarial examples.
- Save artifacts:
  - Save the learned `v` as a `.npy` file and a few example adversarial wavs in an `outputs/` directory for analysis.[1]

## Step 4: Minimal evaluation and research hooks

Even early, plug in evaluation and “novelty hooks” that you can expand later.[1]

- Extend `evaluate.py` to:
  - Compare:
    - Clean vs PGD vs UAP WER/CER on test utterances.
    - Random noise vs adversarial perturbations at the same SNR.[1]
- Add placeholders for:
  - Cross-language evaluation:
    - Functions that test the same universal perturbation on CommonVoice in other languages.[1]
  - Defense-aware evaluation:
    - Functions that add Gaussian noise (randomized smoothing) before Whisper and measure changes in attack success rate.

If you want, the next step can be to translate this into concrete tasks in a project board (e.g., Week 1/2 tickets) and to sketch the core PyTorch pseudocode for `pgd.py` and `uap.py` so you can start implementing right away.
