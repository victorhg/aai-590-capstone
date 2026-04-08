# Q&A: Common Questions & Answers

## General Technical Questions

**Q: What makes Whisper vulnerable to this?**
A: Whisper processes audio through a log-mel spectrogram before the Transformer encoder-decoder. We re-implemented that spectrogram computation in pure PyTorch so gradients flow from the decoder loss all the way back to the raw waveform. This lets us compute exactly which tiny audio changes maximally disrupt (or redirect) the model's output, then apply those changes iteratively.

**Q: Is the attack audible?**
A: It depends on the attack. PGD (per-sample untargeted) achieves 26.58 dB SNR — moderately perceptible, sounds like faint static. The universal targeted attack (UCW) operates at 11.34 dB SNR, which is more noticeable but still sounds like background noise rather than speech. None of our current attacks reach the ideal 35–45 dB imperceptibility range, which is the main open challenge.

**Q: What is "Universal"?**
A: A standard adversarial attack creates a custom perturbation for one specific audio clip. A Universal Adversarial Perturbation (UAP) is a single, fixed noise pattern trained across many samples — it works on *any* audio without re-optimization. Our untargeted UAP fools 60% of unseen utterances; our targeted UCW injects a specific phrase on 58.6% of 350 held-out samples across 40 speakers.

**Q: What is the difference between untargeted and targeted?**
A: Untargeted attacks just corrupt the transcription — garbled text, repetitions, hallucinations. Targeted attacks force the model to output a *specific phrase* of the attacker's choosing, regardless of what was actually said. Targeted is much harder but has more severe security implications.

## Application Questions

**Q: Can this actually be used to attack a real system?**
A: In the digital domain — batch transcription, API processing, embedded audio files — yes. The perturbation can be mixed into audio before it reaches the model. The practical limitation is audibility: at 11 dB SNR, a careful listener may notice the noise. For over-the-air attacks (playing through speakers), additional degradation from room acoustics would reduce effectiveness.

**Q: What about agentic pipelines?**
A: This is the most concerning implication. If Whisper transcription feeds into an LLM pipeline (e.g., a voice assistant that executes commands), the injected phrase becomes a prompt injection at the audio layer. Even a 58.6% success rate across diverse inputs represents a scalable attack vector.

**Q: Why not just train the model to be more robust?**
A: Adversarial training is an active research area, but defending against one attack family does not guarantee robustness against others. Our results show that even Whisper — trained on 680,000 hours of diverse audio — has exploitable directions in its spectrogram space. The vulnerability appears structural to gradient-based models.

**Q: Does this work in real-time?**
A: Yes. The Gradio demo in `notebooks/08_train_demo_targeted_attack.ipynb` supports live streaming with real-time perturbation application. Universal perturbations are applied as simple waveform addition — the bottleneck is Whisper inference, not the attack.

## Project Specifics

**Q: What hardware did you use?**
A: Apple Silicon Mac with MPS backend for PyTorch. Training the UCW perturbation (1,600 samples, up to 200 epochs with gradient accumulation) took several hours. Inference and demo run near real-time.

**Q: Can this attack work on other languages?**
A: The gradient-based mechanism is language-agnostic. We used `openai/whisper-base` (multilingual) and focused on English for this project. Multilingual evaluation is straightforward future work.

**Q: How did you generate the targeted attack?**
A: Two approaches. Per-sample: Carlini-Wagner optimization minimizes perturbation L2 norm + cross-entropy toward target tokens, with binary search over the tradeoff parameter c. Universal: PGD sign-gradient updates trained across 1,600 samples with L∞ constraint (ε = 0.02) and cosine annealing. The target phrase "access evil website" is tokenized and appended to Whisper's decoder prefix for loss computation.

**Q: Why 58.6% and not higher for the universal targeted attack?**
A: The training-to-validation gap (90% train → 58.6% val) reflects partial overfitting. The perturbation learns directions that generalize broadly but not perfectly across all speakers and acoustic conditions. Larger training sets, curriculum learning, or warm-starting from the untargeted UAP are promising paths to improve this.

**Q: Why does the UAP produce garbled text like "the the the"?**
A: The untargeted perturbation is optimized to minimize encoder-state dispersion — it makes the internal representations incoherent. The decoder then struggles to align tokens and often outputs repetitions, hallucinated words, or empty transcriptions.

## Key Numbers Reference

| Attack | Success Rate | Mean SNR | Samples |
|---|---:|---:|---:|
| PGD (per-sample untargeted) | 92% | 26.58 dB | 50 |
| UAP (universal untargeted) | 60% | ~16–18 dB | 20 val |
| CW (per-sample targeted) | Demonstrated | 14.50 dB | 5 |
| UCW (universal targeted) | 58.6% | 11.34 dB | 350 val |
| Baseline clean WER | 4.78% | — | 50 |

## Takeaways

*   Adversarial examples are not limited to images — audio is equally vulnerable.
*   Universal attacks make this a scalable threat: one pre-computed noise file, applied to any audio.
*   The central open challenge is pushing effectiveness into the imperceptible SNR range (35+ dB).
*   Defense mechanisms, over-the-air testing, and cross-model transfer are critical next steps.

