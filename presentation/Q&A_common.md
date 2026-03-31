# Q&A: Common Questions & Answers

## General Technical Questions

**Q: What makes Whisper vulnerable to this?**
A: Whisper uses an `Encoder` that processes audio spectrograms. The optimization loop treats the audio signal as a differentiable tensor. We use backpropagation to calculate which audio frequencies cause the highest probability error in the Decoder, allowing us to iteratively "paint" noise that maximizes this error.

**Q: Is the attack audible?**
A: **No.** This is the most critical constraint. Our perturbation is designed to maintain an SNR (Signal-to-Noise Ratio) of approximately 35-40 dB. This is significantly above the "Just Noticeable Difference" (JND) threshold for human hearing, meaning the listener hears standard background noise, not an obvious "hiss" or "glitch."

**Q: What is "Universal"?**
A: A standard attack might work perfectly on one specific recording but fail on another due to background noise. A Universal Adversarial Perturbation (UAP) is a single, fixed sound file that is trained to fool the model across *any* English audio sample. It transfers well between speakers.

## Application Questions

**Q: Can this actually be used to prank someone or steal data?**
A: Theoretically, yes. In a social engineering scenario, an attacker could send a "sound file" (e.g., a malicious sticker message in a chat app) containing the perturbation. When the victim plays it, their phone (running Whisper) could transcribe it as a specific command, bypassing security checks.

**Q: Why not just train the model to be more robust?**
A: This is an active area of research (Adversarial Training). However, simply adding noise during training doesn't always guarantee robustness against unseen attacks. Furthermore, our attack demonstrates that *any* model (even state-of-the-art ones) has "blind spots" in their spectrogram inputs.

**Q: Does this work in real-time?**
A: Yes, the `src/demo/live_transcribe.py` demonstrates real-time processing. The latency is dominated by the Whisper model inference, which is currently limited by GPU speeds. For real-world application, optimization is needed.

## Project Specifics

**Q: What hardware did you use?**
A: Primarily a local NVIDIA GPU (CUDA) for training the Universal Perturbations and running the model inference. Training took several hours, while inference runs near real-time.

**Q: Can this attack work on other languages?**
A: Yes, but we focused on English (`whisper-base.en`) for this project. The mechanism (Optimization Loop) applies to any model, though the specific optimal noise frequency might differ.

**Q: Is the code ready to use?**
A: Yes. Please refer to the `DEMO_SETUP.md` file for instructions on how to run the live demonstration on your local machine.

**Q: How did you generate the Targeted Attack?**
A: We used a Carlini-Wagner (CW) optimization loop. We treat the problem as an optimization: `minimize Loss(model(audio))` where we *force* the model to output a specific string (e.g., "aai590"). We iterate over the audio samples, calculating gradients, and project the noise onto an $\epsilon$-ball to keep the audio clean.

**Q: Why does the UAP make the transcript "The the the"?**
A: The perturbation is designed to force the model's internal confidence scores to collapse. Instead of predicting "Hello world", the model struggles to align tokens and often outputs repetition or gibberish.

## Conclusion

*   **The Takeaway**: Adversarial examples are not just for images (like the famous cat image); they exist in audio too.
*   **The Threat**: It is easier to create a sound file than a video file.
*   **The Solution**: We need robust testing and defense mechanisms in all AI systems.

