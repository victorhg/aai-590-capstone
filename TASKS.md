# TASKS

## Week 1: Foundation, Setup & EDA

# Environment Setup & Library Installation [NOT DONE]
- [ ] Create Python 3.10+ virtual environment
- [ ] Install `openai-whisper` and `speechbrain`
- [ ] Install `torch` with CUDA/MPS support
- [ ] Install analysis and audio tools: `librosa`, `jiwer`, `soundfile`, `numpy`, `matplotlib`
- [ ] Verify GPU availability and memory access

# Dataset Acquisition & Preprocessing [NOT DONE]
- [ ] Download LibriSpeech `test-clean` subset (~75 utterances)
- [ ] Download CommonVoice sample (10-20 multilingual utterances)
- [ ] Create `src/data` directory structure
- [ ] Implement audio loading script (ensure 16kHz resampling)
- [ ] Implement audio normalization utils (float32, range [-1, 1])

# Exploratory Data Analysis (EDA) [NOT DONE]
- [ ] Create `notebooks/01_explore_dataset.ipynb` for interactive analysis
- [ ] Visualize waveforms and Mel-spectrograms of 5 random LibriSpeech samples
- [ ] Analyze audio duration distribution to determine optimal UAP vector length (e.g., 30s vs max duration)
- [ ] Check amplitude statistics (min, max, mean) to confirm normalization needs
- [ ] Compare frequency markers of "Clean" vs "Noisy" audio (sanity check dummy noise)

## Week 2: Baseline & Untargeted PGD Attack

# Baseline Performance Evaluation [NOT DONE]
- [ ] Create `notebooks/02_performance_evaluation.ipynb` for explanation
- [ ] Run Whisper on clean LibriSpeech dataset
- [ ] Compute and log baseline WER (Word Error Rate) and CER (Character Error Rate)
- [ ] Store baseline transcriptions for reference
- [ ] Verify SNR calculation function against reference implementation (ensure log10 math is correct)

# PGD Attack Implementation [NOT DONE]
- [ ] Create `notebooks/03_pgd_attack.ipynb` for explanation and evidence of work
- [ ] Implement `src/attacks/pgd.py` structure
- [ ] Implement gradient computation loop using PyTorch `autograd`
- [ ] Create wrapper to pass gradients through Whisper's Mel-spectrogram layer
- [ ] Implement `clip` function to enforce $L_\infty$ or $L_2$ norm constraints
- [ ] Implement Optimization loop (iterative noise addition) 

# PGD Experimentation & Tuning [NOT DONE]
- [ ] Run PGD attack on single utterance
- [ ] Tune hyperparameters: learning rate, epsilon, iterations
- [ ] Batch process 10 utterances and record WER/SNR
- [ ] Generate analysis plots: WER vs SNR tradeoff
- [ ] Update the `notebooks/03_pgd_attack.ipynb` with the results

## Week 4-5: Universal Adversarial Perturbations (UAP) - Core

# UAP Training Loop Implementation [NOT DONE]
- [ ] create the `notebooks/04_uap_training.ipynb` with the results
- [ ] Initialize global perturbation vector $v$ (zeros)
- [ ] Implement "Accumulated Gradient" approach over training set
- [ ] Implement `minimize` step for current audio sample $x_i$
- [ ] Implement projection step to keep global perturbation $v$ within $\epsilon$-ball

# UAP Validation & Tuning [NOT DONE]
- [ ] Split LibriSpeech into Train (70) and Validation (20) sets
- [ ] Monitor "Success Rate" (CER > 0.5) during training epochs
- [ ] Tune `regularization_c` and `SNR_target`
- [ ] Save best performing Universal Perturbation vector

## Week 5-6: Evaluation & Defense

# Comprehensive Evaluation [NOT DONE]
- [ ] Create `notebooks/05_defense_evaluation.ipynb` to examplify use
- [ ] Run Universal Perturbation on full Test set (75 utterances)
- [ ] Calculate final metrics: Mean WER, Mean CER, Mean SNR, Success Rate
- [ ] Run Cross-Project evaluation (e.g., test on CommonVoice samples with English perturbation)

# Defense Mechanism Implementation (Randomized Smoothing) [NOT DONE]
- [ ] Implement Gaussian noise injection pre-processor
- [ ] Evaluate defense: Run UAP attack against "smoothed" model
- [ ] Measure drop in Attack Success Rate vs. increase in Clean WER

## Week 6: Reporting

# Project Report & Visualization [NOT DONE]
- [ ] Generate audio samples (Clean vs. Adversarial) for demo
- [ ] Plot final Success Rate vs SNR curves
- [ ] Write technical report documenting methodology and results
- [ ] Create `notebooks/06_Tech_Report.md` for explanation

---

## Experimental / Optional

# Targeted CW Attack (Time Permitting) [NOT DONE]
- [ ] Implement weighted CTC loss for targeted phrases
- [ ] Implement Carlini-Wagner optimization loop ($L_2$ penalty)
- [ ] Test on 5 utterances with specific target phrases