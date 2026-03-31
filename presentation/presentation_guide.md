# Presentation Guide: Adversarial Attacks on ASR

**Topic**: Understanding and Mitigating Vulnerabilities in Automatic Speech Recognition
**Audience**: Computer Science / AI Students / Faculty
**Duration**: 10-15 Minutes

---

## Introduction (2 Minutes)

*   **Hook**: "Have you ever had your smart assistant misunderstand you in a crowded room? Or perhaps a 'friend' pranking you by changing their voice to say something you didn't mean?"
*   **Context**: We live in an era where Speech Recognition (ASR) is ubiquitous (Smart Speakers, Transcription tools).
*   **The Problem**: Is it safe? Can we manipulate ASR systems without the user noticing?
*   **Project Goal**: This project investigates the robustness of OpenAI's Whisper model against two types of adversarial attacks: **Untargeted Corruption** and **Targeted Injection**.

---

## Technical Overview (3 Minutes)

### 1. The Architecture (Whisper)
*   **Visual**: Show a diagram of Whisper (Encoder-Decoder structure).
*   **Explanation**: Whisper takes audio, converts it to Log-Mel Spectrograms, and predicts text using a Transformer-based decoder.
*   **Key Insight**: Because the input (Audio) and the model are differentiable, we can calculate gradients to find the specific sound wave (noise) that confuses the model.

### 2. Methodology
*   **Untargeted Attack**: We add noise to the input audio to maximize the error rate (WER). This is often called "The Jammer".
*   **Targeted Attack**: We add noise to force the model to output a *specific* phrase (e.g., "This is a Demo - aai590").
*   **Techniques Used**:
    *   **PGD (Projected Gradient Descent)**: For iterative optimization.
    *   **CW (Carlini-Wagner)**: For targeted optimization (finding minimal noise to achieve a specific goal).

---

## The Demo (5 Minutes)

*(Perform the live demo here)*

1.  **Clean vs. Adversarial**:
    *   Start with **Clean Mode**. Read a standard sentence (e.g., "The weather is nice today").
    *   *Result*: Accurate transcription.
    *   Switch to **Untargeted UAP Mode**.
    *   Repeat the sentence.
    *   *Result*: **"the the the the..."** or **"unintelligible noise"**.
    *   *Visual*: Show the SNR drop (if UI supports) or simply the corruption.
    *   *Explanation*: This is a Universal Perturbation. It attacks *any* English audio, not just specific recordings.

2.  **Targeted Injection** (If trained):
    *   Switch to **Targeted Mode**.
    *   Say a neutral phrase.
    *   *Result*: The transcript magically changes to the target phrase (e.g., "aai590").

### Discussion during Demo
*   **Imperceptibility**: Did you hear the noise? "No, it sounded like background noise."
*   **Transferability**: Does this attack work on *any* English speaker? Yes, that's why it's "Universal".

---

## Results & Analysis (2 Minutes)

*   **Performance Metrics**:
    *   **WER (Word Error Rate)**: Increased significantly from ~5% (Clean) to ~40%+ (Adversarial).
    *   **SNR (Signal-to-Noise Ratio)**: We maintained SNR > 35dB to ensure the attack was imperceptible to human ears.
*   **Visualization**: Show the Spectrogram of the original vs. the adversarial audio. The adversarial wave looks like white noise, but the model reads it as text.

---

## Conclusion & Q&A (3 Minutes)

### Summary
*   We successfully demonstrated that ASR systems are vulnerable to sound-based attacks.
*   These attacks are invisible to the human ear (high SNR) but catastrophic to the model.
*   **Real-world Implications**: Privacy concerns, social engineering, and misinformation spread via voice messages.

### Future Work
*   Defense mechanisms (Adversarial Training, Input Filtering).
*   Testing on real-world devices (phones, laptops).
*   Multilingual attacks.

### Q&A
*   *Q: Can I just play a recording of this noise?* A: Yes, the perturbation is a fixed sound file. If you play it over a phone call, the receiver's mic picks it up.
*   *Q: How do you hide the noise?* A: We use specific frequencies (white noise) that match the model's blind spots in the Mel-spectrogram input.
*   *Q: Is this code open source?* A: Yes, please see the repository link.

---

## Slides Content Ideas

*   **Slide 1**: Title, Author, University.
*   **Slide 2**: Abstract (Brief summary).
*   **Slide 3**: Problem Statement (ASR Vulnerabilities).
*   **Slide 4**: Technical Deep Dive (How Whisper works).
*   **Slide 5**: The Attack Pipeline (Input -> Perturbation -> Whisper -> Output).
*   **Slide 6**: Demo Results (Bar charts of WER).
*   **Slide 7**: Conclusion.

