
# Audio Adversarial Attacks on ASR (Whisper)

AAI-590 Capstone Project

**Victor Hugo Germano**

_Shiley-Marcos School of Engineering, University of San Diego AAI-500: Probability and Statistics_

## Abstract

The AAI-590 Capstone Project at the University of San Diego in the  M.S. in Applied Artificial Intelligence program. This capstone focus on adversarial attacks against Whisper, OpenAI's automatic speech recognition model. Using multiple adversarial approaches, we implemented Projected Gradient Descent (PGD),  Universal Adversarial Perturbation (UAP) and Targeted Carlini-Wagner (CW), in order to explore Whisper's vulnerabilities to intentional "smart noises" that can affect the model capabilities. The main finding is that Whisper remains vulnerable in the digital white-box setting. Untargeted attacks substantially change transcription output, targeted attacks can force a chosen phrase on a small evaluation batch, and a universal perturbation can generalize across multiple utterances. At the same time, the most successful attacks in the current implementation often operate at SNR levels that are more audible than the original ideal target, so effectiveness and imperceptibility remain the central tradeoff.


## Project objective

Implementing adversarial attacks on OpenAI's Whisper speech recognition model. The project focuses on generating imperceptible audio perturbations that cause transcription failures. This is a critical security issue with implications for content moderation, command injection, and data poisoning. Adversarial attacks on speech recognition exploit a critical vulnerability: while Whisper is robust to natural noise and random perturbations, it is highly vulnerable to adversarial noise where small, crafted modifications are specifically optimized to fool the model.



## Reference

Olivier, R., & Raj, B. (2023). "Fooling Whisper with adversarial examples." *Interspeech 2023*. [Isca-archive](https://www.isca-archive.org/interspeech_2023/olivier23_interspeech.pdf)
- Project reference: https://github.com/RaphaelOlivier/whisper_attack

Olivier, R., & Raj, B. (2022). *There is more than one kind of robustness: Fooling Whisper with adversarial examples* (arXiv:2210.17316). arXiv. [semanticscholar](https://www.semanticscholar.org/paper/There-is-more-than-one-kind-of-robustness:-Fooling-Olivier-Raj/286faebc2be7050c0ab4c049f9db7e9bdf81cbca)

Neekhara, P., Hussain, S., Pandey, P., Dubnov, S., McAuley, J., & Koushanfar, F. (2019). "Universal adversarial perturbations for speech recognition systems." *Interspeech 2019*.

# Tasks overview


## Setup & EDA
- [./notebooks/01_explore_dataset.ipynb](./notebooks/01_explore_dataset.ipynb)
- Environment Setup & Library Installation
-  Dataset Acquisition & Preprocessing (LibriSpeech & commonVoice)
- Exploratory Data Analysis (EDA) 
	- Visualize waveforms and Mel-spectrograms of 5 random LibriSpeech samples


### Baseline &  PGD Attack
- [Baseline Performance Evaluation (notebook)](./notebooks/02_performance_evaluation.ipynb)
	- Run Whisper on clean LibriSpeech dataset
	- Compute and log baseline WER (Word Error Rate) and CER (Character Error Rate)
	- Store baseline transcriptions for reference
	- Verify SNR calculation function against reference implementation (ensure log10 math is correct)
-  [PGD Attack Implementation (notebook)](./notebooks/03_pgd_attack.ipynb)
	- PGD Experimentation & Tuning 
	- Run PGD attack on batch utterance
	- Generate analysis plots: WER vs SNR tradeoff

## Universal Adversarial Perturbations (UAP) 

- UAP Training Loop Implementation 
- UAP Validation & Tuning

## Evaluation & Defense

- Run Universal Perturbation on full Test set (75 utterances)
	- Calculate final metrics: Mean WER, Mean CER, Mean SNR, Success Rate
	- Run Cross-Project evaluation (e.g., test on CommonVoice samples with English perturbation)


## Reporting

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

