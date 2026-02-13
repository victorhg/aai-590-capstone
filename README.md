
# Audio Adversarial Attacks on ASR (Whisper)

AAI-590 Capstone Project

**Victor Hugo Germano**

_Shiley-Marcos School of Engineering, University of San Diego AAI-500: Probability and Statistics_


## Project objective

Implementing adversarial attacks on OpenAI's Whisper speech recognition model. The project focuses on generating imperceptible audio perturbations that cause transcription failures. This is a critical security issue with implications for content moderation, command injection, and data poisoning. Adversarial attacks on speech recognition exploit a critical vulnerability: while Whisper is robust to natural noise and random perturbations, it is highly vulnerable to adversarial noise where small, crafted modifications are specifically optimized to fool the model.


## Reference

Olivier, R., & Raj, B. (2023). "Fooling Whisper with adversarial examples." *Interspeech 2023*. [Isca-archive](https://www.isca-archive.org/interspeech_2023/olivier23_interspeech.pdf)
- Project reference: https://github.com/RaphaelOlivier/whisper_attack

Olivier, R., & Raj, B. (2022). *There is more than one kind of robustness: Fooling Whisper with adversarial examples* (arXiv:2210.17316). arXiv. [semanticscholar](https://www.semanticscholar.org/paper/There-is-more-than-one-kind-of-robustness:-Fooling-Olivier-Raj/286faebc2be7050c0ab4c049f9db7e9bdf81cbca)

Neekhara, P., Hussain, S., Pandey, P., Dubnov, S., McAuley, J., & Koushanfar, F. (2019). "Universal adversarial perturbations for speech recognition systems." *Interspeech 2019*.

# Tasks overview


## Week 1: Setup & EDA

- Environment Setup & Library Installation
-  Dataset Acquisition & Preprocessing (LibriSpeech & commonVoice)
- Exploratory Data Analysis (EDA) 
	- Visualize waveforms and Mel-spectrograms of 5 random LibriSpeech samples


### Week 2: Baseline &  PGD Attack

- Baseline Performance Evaluation
	- Run Whisper on clean LibriSpeech dataset
	- Compute and log baseline WER (Word Error Rate) and CER (Character Error Rate)
	- Store baseline transcriptions for reference
	- Verify SNR calculation function against reference implementation (ensure log10 math is correct)
-  PGD Attack Implementation
	- PGD Experimentation & Tuning 
	- Run PGD attack on batch utterance
	- Generate analysis plots: WER vs SNR tradeoff

### Week 4-5: Universal Adversarial Perturbations (UAP) 

- UAP Training Loop Implementation 
- UAP Validation & Tuning

### Week 5-6: Evaluation & Defense

- Run Universal Perturbation on full Test set (75 utterances)
	- Calculate final metrics: Mean WER, Mean CER, Mean SNR, Success Rate
	- Run Cross-Project evaluation (e.g., test on CommonVoice samples with English perturbation)


## Week 6-7: Reporting

- Project Report & Visualization 
- Generate audio samples (Clean vs. Adversarial) for demo
- Plot final Success Rate vs SNR curves


---

## Experimental / Optional

- Real time live demonstration
	- live transcript being affected by sound
- Defense Mechanism Implementation (Randomized Smoothing) 
	- Implement Gaussian noise injection pre-processor
	- Evaluate defense: Run UAP attack against "smoothed" model
	- Measure drop in Attack Success Rate vs. increase in Clean WER

- Targeted CW Attack 
	- Implement weighted CTC loss for targeted phrases
	- Test on 5 utterances with specific target phrases

