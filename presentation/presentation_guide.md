# Presentation Guide: Adversarial Attacks on Whisper ASR

**Topic**: Audio Adversarial Attacks on OpenAI Whisper
**Author**: Victor Hugo Germano — AAI-590 Capstone
**Audience**: Computer Science / AI Students / Faculty
**Duration**: 10–15 Minutes

---

## Introduction (2 Minutes)

*   **Hook**: "Whisper was trained on 680,000 hours of audio. It handles accents, background noise, and low-quality recordings. But what happens when the noise is *designed* to fool it?"
*   **Context**: ASR is now embedded in voice assistants, transcription services, accessibility tools, and agentic LLM pipelines. Whisper is one of the most widely deployed open models.
*   **The Problem**: If an attacker can add a small, carefully crafted perturbation to audio, ASR output can be degraded or hijacked entirely — even when the perturbation sounds like faint static to a human.
*   **Project Goal**: We implemented and evaluated **four attack families** against Whisper-base in a white-box digital setting:
    1.  Per-sample untargeted (PGD)
    2.  Universal untargeted (UAP — the "digital jammer")
    3.  Per-sample targeted (CW — phrase injection)
    4.  Universal targeted (UCW — scalable phrase injection)

---

## Technical Overview (3 Minutes)

### 1. Whisper Architecture
*   **Visual**: Show a diagram of Whisper's Encoder-Decoder pipeline.
*   **Key path**: Raw audio → Log-Mel Spectrogram (80 bins, 3,000 frames) → Transformer Encoder → Transformer Decoder → Text tokens.
*   **Critical insight**: We re-implemented the mel-spectrogram computation in pure PyTorch (STFT → power → mel → log) so gradients flow all the way back from the decoder loss to the raw waveform. This is what makes whitebox attacks possible.

### 2. The Four Attacks

| Attack | What it does | Key idea |
|---|---|---|
| **PGD** | Corrupts one audio clip at a time | Signed gradient steps + L∞ projection |
| **UAP** | One fixed perturbation degrades *any* audio | Train across many samples, skip already-fooled ones |
| **CW** | Forces a specific phrase on one clip | Minimize L2 norm + cross-entropy toward target tokens |
| **UCW** | One fixed perturbation forces a phrase on *any* audio | PGD sign-gradient over 1,600 training files with cosine annealing |

*   **Perturbation equation**: $x_{adv} = \text{clamp}(x + \delta,\ -1,\ 1)$ where $\delta$ is bounded by $\|\delta\|_\infty \le \epsilon$.
*   **Loss modes**: Untargeted minimizes encoder-state dispersion; Targeted uses cross-entropy with Whisper's decoder prefix + target tokens.

---

## The Demo (5 Minutes)

*(Launch `notebooks/08_train_demo_targeted_attack.ipynb` — Gradio interface)*

### Part 1: Single-Audio Demo

1.  **Clean Mode**: Upload or record a sentence (e.g., "The weather is nice today").
    *   *Expected*: Near-perfect transcription (baseline WER ≈ 4.78%).
2.  **Untargeted UAP Mode**: Same audio, now with the universal perturbation overlaid.
    *   *Expected*: Corrupted transcript — repetitions, garbled words, or hallucinated text.
    *   *Show*: The SNR readout and waveform comparison plot.
    *   *Explain*: "This is a single, fixed noise pattern that was trained once and works against any English audio. It's a digital jammer."
3.  **Targeted CW Injection Mode**: Same audio, now with the UCW perturbation.
    *   *Expected*: Transcript changes to **"access evil website"** regardless of what was actually said.
    *   *Show*: Side-by-side clean vs. adversarial transcript.
    *   *Explain*: "This perturbation was trained on 1,600 utterances to universally inject a target phrase. On held-out validation, it succeeded on 58.6% of 350 unseen utterances."

### Part 2: Live Streaming Demo

1.  **Clean Mode**: Speak into the microphone. Watch real-time transcription.
2.  **UAP Mode**: Same speech, live corruption. Show the history panel accumulating garbled text.
3.  **Targeted Mode**: Speak normally. After a few seconds of accumulated audio, the committed transcript should show the target phrase.
    *   *Note*: The streaming system accumulates ≥3 seconds of speech before applying the targeted perturbation, because short fragments don't contain enough mel-spectrogram coverage.

### Discussion Points During Demo
*   **Audibility**: "Can you hear a difference?" — The perturbation sounds like faint static. At 26 dB SNR (PGD) it is barely noticeable; at 11 dB (UCW) it is more audible but still sounds like background noise, not speech.
*   **Universality**: "This exact same noise file works on any speaker, any sentence. It was pre-computed offline."
*   **Agentic threat**: "If this transcription feeds into an LLM pipeline, the injected phrase becomes a prompt injection at the audio layer."

---

## Results & Analysis (2 Minutes)

### Baseline
*   Whisper-base on LibriSpeech test-clean: **WER 4.78%, CER 1.98%** (50 samples).

### Attack Results Summary

| Attack | Success Rate | Mean SNR | Key Detail |
|---|---:|---:|---|
| **PGD** (per-sample untargeted) | **92%** (46/50) | **26.58 dB** | ε = 0.01, 20 iterations |
| **UAP** (universal untargeted) | **60%** (12/20 val) | ~16–18 dB | 5 s perturbation, ε = 0.08 |
| **CW** (per-sample targeted) | Demonstrated | **14.50 dB** | Target: "hello world", c = 50 |
| **UCW** (universal targeted) | **58.6%** (205/350 val) | **11.34 dB** | Target: "access evil website", ε = 0.02 |

### Key Takeaways
*   **PGD** is the most effective per-sample attack — 92% success with moderate audibility.
*   **UAP** generalizes to unseen speakers — a single file degrades 60% of utterances.
*   **UCW** is the headline result — a fixed perturbation forces a specific phrase on 58.6% of 350 unseen utterances across 40 speakers.
*   **Tradeoff**: Per-sample attacks achieve better SNR (less audible) but require model access at attack time. Universal attacks sacrifice SNR for deployability — pre-compute once, apply anywhere.
*   **SNR gap**: All attacks fall below the ideal 35–45 dB imperceptibility range. Improving this is the primary open challenge.

---

## Conclusion & Q&A (3 Minutes)

### Summary
*   Whisper-base is vulnerable to all four attack modes despite being trained on 680,000 hours of audio.
*   Universal attacks (UAP, UCW) are the most operationally significant: a single pre-computed perturbation works across diverse speakers without re-optimization.
*   The UCW result — injecting "access evil website" on 58.6% of unseen utterances — demonstrates a scalable audio-layer prompt-injection vector.
*   Current limitation: SNR levels (11–27 dB) make perturbations partially audible. Pushing effectiveness into the 35+ dB range is future work.

### Future Work
*   **Psychoacoustic masking** — perceptually informed loss to hide perturbations in spectral regions humans can't hear.
*   **Over-the-air testing** — do perturbations survive physical playback and re-recording?
*   **Black-box transfer** — do these perturbations fool wav2vec 2.0 or Whisper-large?
*   **Multilingual attacks** — extend beyond English.

### Anticipated Q&A
*   *Q: Can I just play this noise file over a speaker?* → Theoretically yes, but over-the-air degradation would reduce effectiveness. This project tests the digital domain.
*   *Q: How do you hide the noise?* → The perturbation exploits specific frequency bins in Whisper's mel-spectrogram. To a human, it sounds like faint static.
*   *Q: Why not just train Whisper to be robust?* → Adversarial training is an active research area, but it doesn't guarantee robustness against unseen attack families. Our results show the vulnerability persists even in a well-trained model.
*   *Q: Does this work on other languages?* → The mechanism applies to any language Whisper supports, but we focused on English for this project.

---

## Slide Outline

| Slide | Content |
|---|---|
| 1 | Title — Audio Adversarial Attacks on Whisper. Author, University. |
| 2 | Problem — ASR is everywhere; adversarial robustness is a security question. |
| 3 | Whisper Architecture — Encoder-Decoder diagram, mel-spectrogram pipeline. |
| 4 | The Four Attacks — Table with PGD / UAP / CW / UCW and what each does. |
| 5 | Attack Pipeline — Diagram: Audio → +δ → Mel → Whisper → Corrupted/Injected text. |
| 6 | **LIVE DEMO** — Switch to Gradio. Clean → UAP → Targeted. |
| 7 | Results Table — Success rates, SNR, cross-attack comparison. |
| 8 | UCW Deep Dive — Training curve (0% → 90% train, 58.6% val), design evolution. |
| 9 | Discussion — Tradeoffs, SNR gap, agentic implications. |
| 10 | Future Work & Conclusion — Psychoacoustic masking, over-the-air, transfer. |

