# TASKS

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

# Untargeted Attack Integration (Demo Part 1: Corruption) [DONE]
- [x] Load pre-trained UAP vector from `notebooks/04_uap_training.ipynb` or `06_uap_jammer.ipynb`
- [x] Save UAP as standalone file: `models/trained_uap.pt`
- [x] Implement real-time UAP application function in `src/demo/attack_utils.py`
- [x] Test UAP on streaming audio chunks (verify differentiability not required for inference)
- [x] Verify imperceptibility: measure SNR on test samples (target: >35dB)
- [x] Create demo script: record yourself → apply UAP → show corrupted transcript

# Targeted Attack Training for Demo Phrase (Demo Part 2: Injection) [DONE]
- [x] Create `notebooks/08_train_demo_targeted_attack.ipynb` for training documentation
- [ ] Record yourself saying a neutral phrase (e.g., "Welcome to my demonstration")
- [ ] Save as `demo_assets/my_voice_generic.wav`
- [x] Train Carlini-Wagner attack to inject: "This is a Demo - aai590"
- [x] Tune CW hyperparameters: epsilon=0.03-0.05, iterations=1500-2500, learning_rate=0.01
- [x] Verify injection success: original transcript vs. adversarial transcript
- [x] Save trained perturbation as `demo_assets/targeted_perturbation.pt`
- [x] Measure and log SNR for imperceptibility validation

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

# Documentation & Demo Materials [DONE]
- [x] Create `DEMO_SETUP.md` with installation instructions
- [x] Document demo script with timing and talking points
- [x] Prepare slide deck explaining: Differentiability → Gradients → Attacks
- [x] Add troubleshooting section for common issues (microphone permissions, latency)
- [x] Create README section highlighting the demo capabilities
- [x] Prepare Q&A responses for common questions (transferability, defenses, real-world impact)
