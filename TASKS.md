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

# Technical report [DONE]
- [x] Create `notebooks/07_Tech_Report.md` with a general view of the arquitecture and the technical decisions made in this project

---

## Experimental / Optional

# Targeted CW Attack (Time Permitting) [DONE]
- [x] Implement weighted CTC loss for targeted phrases
- [x] Implement Carlini-Wagner optimization loop ($L_2$ penalty)
- [x] Test on 5 utterances with specific target phrases




## Week 7-8: Live Demonstration System

# Real-time Audio Infrastructure [DONE]
- [x] Install real-time audio libraries (`pyaudio`, `sounddevice`, `queue`)
- [x] Create `src/demo/audio_stream.py` for microphone capture
- [x] Implement circular buffer for 30-second audio segments (Whisper requirement)
- [x] Implement audio chunk accumulation with overlap handling
- [x] Test microphone input quality and verify 16kHz sampling rate
- [x] Add audio volume normalization for consistent input levels

# Live Transcription UI System [DONE]
- [x] Install UI framework (`gradio` recommended for rapid prototyping)
- [x] Create `src/demo/live_transcribe.py` for main demo application
- [x] Implement clean transcription mode (baseline Whisper)
- [x] Add real-time transcript display with auto-scrolling
- [x] Implement mode selector: [Clean / Untargeted UAP / Targeted Attack]
- [x] Add visual indicators for attack mode status
- [x] Display current SNR and WER metrics in real-time

# Pre-recorded Demo Assets (Fallback Option) [NOT DONE]
- [ ] Record personal voice sample explaining the demo (30-60 seconds)
- [ ] Save as `demo_assets/narrator_clean.wav` at 16kHz
- [ ] Generate baseline transcription and save to `demo_assets/baseline_transcript.txt`
- [ ] Create side-by-side comparison template for video editing
- [ ] Prepare audio visualization plots (waveform + spectrogram)

# Untargeted Attack Integration (Demo Part 1: Corruption) [NOT DONE]
- [ ] Load pre-trained UAP vector from `notebooks/04_uap_training.ipynb` or `06_uap_jammer.ipynb`
- [ ] Save UAP as standalone file: `models/trained_uap.pt`
- [ ] Implement real-time UAP application function in `src/demo/attack_utils.py`
- [ ] Test UAP on streaming audio chunks (verify differentiability not required for inference)
- [ ] Verify imperceptibility: measure SNR on test samples (target: >35dB)
- [ ] Create demo script: record yourself → apply UAP → show corrupted transcript

# Targeted Attack Training for Demo Phrase (Demo Part 2: Injection) [NOT DONE]
- [ ] Create `notebooks/08_train_demo_targeted_attack.ipynb` for training documentation
- [ ] Record yourself saying a neutral phrase (e.g., "Welcome to my demonstration")
- [ ] Save as `demo_assets/my_voice_generic.wav`
- [ ] Train Carlini-Wagner attack to inject: "This is a Demo - aai590"
- [ ] Tune CW hyperparameters: epsilon=0.03-0.05, iterations=1500-2500, learning_rate=0.01
- [ ] Verify injection success: original transcript vs. adversarial transcript
- [ ] Save trained perturbation as `demo_assets/targeted_perturbation.pt`
- [ ] Measure and log SNR for imperceptibility validation

# Demo Integration & Testing [NOT DONE]
- [ ] Integrate targeted attack into live UI as third mode option
- [ ] Implement perturbation loading on application startup (avoid re-computation)
- [ ] Add "Record Demo Audio" button to save attack examples
- [ ] Test full pipeline: Microphone → [Clean/UAP/Targeted] → Whisper → Display
- [ ] Measure end-to-end latency (target: <5 seconds for 30s segment)
- [ ] Create demo rehearsal script with talking points
- [ ] Test on different microphone hardware (laptop built-in vs. external)
- [ ] Prepare backup pre-recorded demo if real-time fails

# Demo Video Production [NOT DONE]
- [ ] Record clean transcription demo (baseline)
- [ ] Record untargeted UAP corruption demo (show WER degradation)
- [ ] Record targeted injection demo (show "aai590" phrase appearing)
- [ ] Capture screen recordings with audio
- [ ] Edit video to show side-by-side comparisons
- [ ] Add annotations/captions explaining each attack stage
- [ ] Export final demo video and upload to presentation platform

# Documentation & Demo Materials [NOT DONE]
- [ ] Create `DEMO_SETUP.md` with installation instructions
- [ ] Document demo script with timing and talking points
- [ ] Prepare slide deck explaining: Differentiability → Gradients → Attacks
- [ ] Add troubleshooting section for common issues (microphone permissions, latency)
- [ ] Create README section highlighting the demo capabilities
- [ ] Prepare Q&A responses for common questions (transferability, defenses, real-world impact)
