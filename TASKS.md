# TASKS

## Week 1: Foundation, Setup & EDA

# Environment Setup & Library Installation [DONE]
- [x] Create Python 3.10+ virtual environment
- [x] Install `openai-whisper` and `speechbrain`
- [x] Install `torch` with CUDA/MPS support
- [x] Install analysis and audio tools: `librosa`, `jiwer`, `soundfile`, `numpy`, `matplotlib`
- [x] Verify GPU availability and memory access

# Dataset Acquisition & Preprocessing [DONE]
- [x] Download LibriSpeech `test-clean` subset (~75 utterances)
- [x] Download CommonVoice sample (10-20 multilingual utterances)
- [x] Create `src/data` directory structure
- [x] Implement audio loading script (ensure 16kHz resampling)
- [x] Implement audio normalization utils (float32, range [-1, 1])

# Exploratory Data Analysis (EDA) [DONE]
- [x] Create `notebooks/01_explore_dataset.ipynb` for interactive analysis
- [x] Visualize waveforms and Mel-spectrograms of 5 random LibriSpeech samples
- [x] Analyze audio duration distribution to determine optimal UAP vector length (e.g., 30s vs max duration)
- [x] Check amplitude statistics (min, max, mean) to confirm normalization needs
- [x] Compare frequency markers of "Clean" vs "Noisy" audio (sanity check dummy noise)

## Week 2: Baseline & Untargeted PGD Attack

# Baseline Performance Evaluation [DONE]
- [x] Create `notebooks/02_performance_evaluation.ipynb` for explanation
- [x] Run Whisper on clean LibriSpeech dataset
- [x] Compute and log baseline WER (Word Error Rate) and CER (Character Error Rate)
- [x] Store baseline transcriptions for reference
- [x] Verify SNR calculation function against reference implementation (ensure log10 math is correct)

# PGD Attack Implementation [DONE]
- [x] Create `notebooks/03_pgd_attack.ipynb` for explanation and evidence of work
- [x] Implement `src/attacks/pgd.py` structure
- [x] Implement gradient computation loop using PyTorch `autograd`
- [x] Create wrapper to pass gradients through Whisper's Mel-spectrogram layer
- [x] Implement `clip` function to enforce $L_\infty$ or $L_2$ norm constraints
- [x] Implement Optimization loop (iterative noise addition) 

# PGD Experimentation & Tuning [DONE]
- [x] Run PGD attack on single utterance
- [x] Tune hyperparameters: learning rate, epsilon, iterations
- [x] Batch process 10 utterances and record WER/SNR
- [x] Generate analysis plots: WER vs SNR tradeoff
- [x] Update the `notebooks/03_pgd_attack.ipynb` with the results

## Week 4-5: Universal Adversarial Perturbations (UAP) - Core

# UAP Training Loop Implementation [DONE]
- [x] create the `notebooks/04_uap_training.ipynb` with the results
- [x] Initialize global perturbation vector $v$ (zeros)
- [x] Implement "Accumulated Gradient" approach over training set
- [x] Implement `minimize` step for current audio sample $x_i$
- [x] Implement projection step to keep global perturbation $v$ within $\epsilon$-ball

# UAP Validation & Tuning [DONE]
- [x] Split LibriSpeech into Train (70) and Validation (20) sets
- [x] Monitor "Success Rate" (CER > 0.5) during training epochs
- [x] Tune `regularization_c` and `SNR_target`
- [x] Save best performing Universal Perturbation vector

## Week 5-6: Evaluation & Defense

# Comprehensive Evaluation [DONE]
- [x] Create `notebooks/05_defense_evaluation.ipynb` to examplify use
- [x] Run Universal Perturbation on full Test set (75 utterances)
- [x] Calculate final metrics: Mean WER, Mean CER, Mean SNR, Success Rate
- [x] Run Cross-Project evaluation (e.g., test on CommonVoice samples with English perturbation)

# Defense Mechanism Implementation (Randomized Smoothing) [DONE]
- [x] Implement Gaussian noise injection pre-processor
- [x] Evaluate defense: Run UAP attack against "smoothed" model
- [x] Measure drop in Attack Success Rate vs. increase in Clean WER

## Week 6: Reporting

# Project Report & Visualization [DONE]
- [x] Generate audio samples (Clean vs. Adversarial) for demo
- [x] Plot final Success Rate vs SNR curves
- [x] Write technical report documenting methodology and results
- [x] Create `notebooks/06_Tech_Report.md` for explanation

---

## Experimental / Optional

# Targeted CW Attack (Time Permitting) [NOT DONE]
- [ ] Implement weighted CTC loss for targeted phrases
- [ ] Implement Carlini-Wagner optimization loop ($L_2$ penalty)
- [ ] Test on 5 utterances with specific target phrases
