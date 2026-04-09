# Teleprompter Script

> Read this script naturally. Pause at `[PAUSE]`. Stage directions are in **[BRACKETS]**.
> Approximate pace: 130 words/minute. Total: ~12 minutes + demo time.

---

## SLIDE 1 — Title (15 seconds)

Hello everyone. My name is Victor Hugo Germano, and today I'm presenting my capstone project: Audio Adversarial Attacks on OpenAI Whisper.

---

## SLIDE 2 — Problem Statement (1 minute 30 seconds)

Whisper is one of the most capable open speech recognition models available today. OpenAI trained it on 680,000 hours of audio, and it handles accents, background noise, and low-quality recordings remarkably well.

[PAUSE]

But here is the question this project investigates: what happens when the noise in the audio is not random — when it is *designed* to fool the model?

[PAUSE]

If I can take any audio file, add a small perturbation — something that sounds like faint static to you — and the model produces a completely wrong transcription, that is a real security vulnerability. And if I can make the model output a *specific phrase* of my choosing, that becomes even more dangerous — especially as transcription systems feed into LLM pipelines and agentic workflows.

[PAUSE]

This project tests that vulnerability with four different attack strategies, progressing from the simplest to the most challenging.

---

## SLIDE 3 — Whisper Architecture (1 minute)

Let me quickly walk through how Whisper works, because the architecture is what makes the attack possible.

Whisper takes raw audio, converts it to a log-mel spectrogram — that is, a visual representation of frequency content over time — and feeds it through a Transformer encoder-decoder to produce text tokens.

The critical technical point is this: in our implementation, the entire pipeline from raw audio to spectrogram to text is differentiable. We re-implemented the mel-spectrogram computation in pure PyTorch — the short-time Fourier transform, the power spectrum, mel filtering, and log normalization — all using operations that support gradient flow.

[PAUSE]

That means we can compute: "which tiny changes to the raw audio would maximally change the model's output?" And then we make exactly those changes. That is the core idea behind all four attacks.

---

## SLIDE 4 — The Four Attacks (1 minute 30 seconds)

We implemented four attack families, organized along two axes: per-sample versus universal, and untargeted versus targeted.

[PAUSE]

**First, PGD — Projected Gradient Descent.** This creates a custom perturbation for a single audio clip. It uses signed gradient steps bounded by an L-infinity constraint. On 50 test samples, PGD achieved a 92 percent success rate — meaning 46 out of 50 transcriptions were measurably corrupted — at an average SNR of 26.58 decibels.

**Second, UAP — Universal Adversarial Perturbation.** This trains a single, fixed perturbation across many audio samples. Once trained, it works on *any* audio without re-optimization. Think of it as a digital jammer. Our UAP achieved a 60 percent fooling rate on held-out samples.

**Third, Carlini-Wagner targeted attack.** Instead of just corrupting the output, this forces the model to produce a specific phrase — in our case, "hello world." This works per-sample and achieved an average SNR of 14.50 dB.

**Fourth, and the headline result — Universal Carlini-Wagner.** This combines the universal approach with targeted injection. A single pre-computed perturbation forces the phrase "access evil website" on any input audio. On 350 unseen validation utterances, it succeeded 58.6 percent of the time.

---

## SLIDE 5 — Attack Pipeline Diagram (30 seconds)

Here is the pipeline visually. Audio comes in. The perturbation delta is added. The combined signal passes through our differentiable mel-spectrogram, into the frozen Whisper encoder and decoder, and produces a loss. We backpropagate that loss to update delta — and only delta. Whisper's weights are never modified.

[PAUSE]

For universal attacks, this loop runs across hundreds or thousands of samples. The result is a single perturbation file that can be applied to any new audio at inference time.

---

## SLIDE 6 — LIVE DEMO (4–5 minutes)

**[SWITCH TO GRADIO INTERFACE]**

Now let me show you this in action.

[PAUSE]

**[SELECT CLEAN MODE]**

I will start in clean mode. I am going to say a sentence, and you will see Whisper transcribe it accurately. This is our baseline — Whisper-base achieves about 4.78 percent word error rate on clean LibriSpeech audio.

**[SPEAK A SENTENCE — e.g., "The weather is really nice today"]**

There you go — accurate transcription.

[PAUSE]

**[SWITCH TO UNTARGETED UAP MODE]**

Now I am switching to untargeted mode. This overlays our universal perturbation — a 5-second noise pattern that is tiled across the audio. Same sentence.

**[SPEAK THE SAME SENTENCE]**

Look at the output. The transcription is garbled. Repetitions, hallucinated words, or gibberish. This is the jammer effect. One fixed noise pattern, and Whisper cannot transcribe correctly.

[PAUSE]

**[SWITCH TO TARGETED CW INJECTION MODE]**

Now the most interesting part. I am switching to targeted mode. This applies our universal targeted perturbation — trained to inject the phrase "access evil website."

I am going to say something completely unrelated.

**[SPEAK A NEUTRAL SENTENCE — e.g., "I had coffee for breakfast this morning"]**

**[WAIT FOR COMMITTED TRANSCRIPT]**

And there it is. Regardless of what I said, the model outputs the target phrase. This is a universal targeted attack — one pre-computed perturbation, no per-sample optimization needed.

[PAUSE]

A few things to notice. Could you hear any difference? The audio sounds like it has some faint static. That is the perturbation. At around 11 dB SNR for the targeted attack, it is partially audible but does not sound like speech. And this same perturbation works on any speaker — it was validated on 40 different speakers.

**[SWITCH BACK TO SLIDES]**

---

## SLIDE 7 — Results Table (1 minute)

Let me put all the numbers side by side.

PGD: 92 percent success, 26.58 dB SNR, but requires per-sample optimization.
UAP: 60 percent success, lower SNR, but completely reusable.
CW targeted: demonstrated phrase injection at 14.50 dB SNR.
UCW: 58.6 percent success on 350 validation samples at 11.34 dB SNR.

[PAUSE]

The clear pattern is: per-sample attacks get higher success rates and better SNR, but they require model access at attack time. Universal attacks sacrifice some effectiveness for deployability — compute once, use everywhere.

---

## SLIDE 8 — UCW Deep Dive (1 minute)

The UCW result is worth explaining further because it was the most technically challenging part of the project.

We trained on 1,600 LibriSpeech utterances across 40 speakers. Training success rate climbed from zero to about 90 percent over 200 epochs. But on 350 held-out validation utterances, it drops to 58.6 percent. That gap tells us the perturbation partially overfits to the training distribution.

[PAUSE]

Several design decisions were critical. We switched from Adam to PGD sign-gradient updates because Adam's adaptive step sizes are misaligned with L-infinity constraints. We removed the L2 penalty because it was consuming roughly half the gradient budget and pushing the perturbation toward zero without improving convergence. And we added cosine annealing to prevent late-stage oscillation.

The perturbation is 10 seconds long and saturates the epsilon boundary at plus or minus 0.02. This gives it enough mel-spectrogram coverage to redirect the decoder.

---

## SLIDE 9 — Discussion (1 minute)

So what does this mean practically?

[PAUSE]

First, Whisper's robustness to natural noise does not extend to adversarial noise. Despite massive pretraining, the model has exploitable directions in its spectrogram space.

Second, universality is the key operational factor. A reusable perturbation that corrupts 60 percent of utterances is a viable jammer. A reusable perturbation that injects a chosen phrase on 58 percent of utterances is a potential prompt-injection vector — at the audio layer, before the text even reaches an LLM.

[PAUSE]

The main limitation is imperceptibility. None of our attacks fully reach the ideal 35 to 45 dB SNR range for truly imperceptible noise. The targeted attacks are the most audible. This means the vulnerability is real for digital pipelines and batch transcription, but not yet a covert threat under close human listening.

---

## SLIDE 10 — Future Work & Conclusion (45 seconds)

Three directions for future work.

First, psychoacoustic masking — using perceptual models to hide perturbations in frequency ranges humans cannot hear. This could push SNR into the imperceptible range.

Second, over-the-air testing — can these perturbations survive physical playback through a speaker and re-recording through a microphone?

Third, transfer attacks — do perturbations trained on Whisper-base also fool Whisper-large or wav2vec 2.0?

[PAUSE]

To conclude: we showed that Whisper-base can be corrupted per-sample at 92 percent success, jammed universally at 60 percent, and forced to output a specific phrase on 58.6 percent of unseen utterances. As ASR becomes the input layer for agentic systems, these vulnerabilities need to be taken seriously.

Thank you. I am happy to take questions.

---

## Q&A REFERENCE (Do not read — use as reference)

**"Can I just play this noise over a speaker?"**
→ In theory yes. But over-the-air degradation — room acoustics, microphone response — would reduce effectiveness. This project tests the digital domain. Over-the-air is future work.

**"How do you hide the noise?"**
→ The perturbation exploits specific frequency bins in Whisper's 80-bin mel-spectrogram. To a human, it sounds like faint static, not speech. At higher SNR values, it would be even less noticeable.

**"Why not train Whisper to resist this?"**
→ Adversarial training is possible but expensive, and defending against one attack family does not guarantee robustness against others. Our results show the vulnerability is structural to the model's spectrogram processing.

**"Does this work on other languages?"**
→ The mechanism is language-agnostic — gradients work the same way. We scoped to English for this project, but multilingual evaluation is straightforward future work.

**"What hardware did you use?"**
→ Training ran on a Mac with Apple Silicon (MPS backend). The UCW training loop processes one sample at a time with gradient accumulation for effective batch size of 4. Training the universal targeted perturbation took several hours.

**"Why 58.6% and not higher?"**
→ The training-to-validation gap (90% train vs 58.6% val) indicates some overfitting. Larger and more diverse training sets, curriculum learning, or warm-starting from the untargeted UAP are plausible paths to higher generalization.

**"Is this a real threat?"**
→ In the digital domain — batch transcription, API-based processing, phone recordings — yes. The perturbation can be embedded in audio files before they reach the model. The main mitigation gap is imperceptibility: at 11 dB SNR a careful listener can detect the noise.
