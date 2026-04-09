AAI-590 - **Victor Hugo Germano**
Shiley-Marcos School of Engineering

# Abstract

The AAI-590 Capstone Project at the University of San Diego in the  M.S. in Applied Artificial Intelligence program. This capstone focus on adversarial attacks against Whisper, OpenAI's automatic speech recognition model. Using multiple adversarial approaches, we implemented Projected Gradient Descent (PGD),  Universal Adversarial Perturbation (UAP) and Targeted Carlini-Wagner (CW), in order to explore Whisper's vulnerabilities to intentional "smart noises" that can affect the model capabilities. The main finding is that Whisper remains vulnerable in the digital white-box setting. Untargeted attacks substantially change transcription output, targeted attacks can force a chosen phrase on a small evaluation batch, and a universal perturbation can generalize across multiple utterances. At the same time, the most successful attacks in the current implementation often operate at SNR levels that are more audible than the original ideal target, so effectiveness and imperceptibility remain the central tradeoff.

# Introduction

As organizations increase their reliance on voice-based systems to improve their relationship with customer and users, it is important to notice that although voice transcription capabilities can work well, there are multiple unknowns about how the technology can be exploited in order to change it's intended behavior. As LLMs become ubiquitous intermediaries between our own interactions, our trusted systems need to adapt to understand new attack vectors and the possibility of real threats (Olivier & Raj 2023). 

Adversarial attacks on Automatic Speech Recognition(ASR) systems involve creating specific audio inputs that cause speech recognition models to produce incorrect transcriptions. These attacks exploit vulnerabilities in deep neural networks by adding designed small modifications to audio signals that are often imperceptible or barely noticeable to human listeners but cause the model to misclassify or mistranscribe the input. This work intends to use these perturbations to affect the model's accuracy and overall ability to reliably transcribe audio. 

Adversarial robustness becomes a practical security question, not just an academic one. If an attacker can add a carefully crafted perturbation $\delta$ to clean audio $x$, then the ASR model may produce a degraded or entirely different transcription:
$$
x_{adv} = \operatorname{clip}(x + \delta, -1, 1)
$$
Ultimately this project wants answers the questions: Can ASR systems really be trusted with our critical scenario information?  Can this technology be secure enough from the intervention of external actors? 

Understanding vulnerabilities in ASR systems is critical for model development (Olivier & Raj 2022) because it enables security applications that prevent malicious actors from bypassing voice authentication systems, supports privacy protection by clarifying how to safeguard sensitive speech data from manipulation, drives robustness improvement through the development of more resilient ASR models that can withstand adversarial perturbations, and addresses safety and privacy considerations by ensuring that voice-controlled systems (such as smart home devices and automotive controls) cannot be hijacked through audio manipulation.

Whisper is OpenAI's automatic speech recognition system (OpenAI 2021), trained on 680,000 hours of multilingual and multitask supervised data collected from the web. 

Whisper's encoder-decoder Transformer Architecture uses an input audio split into 30-second chunks, converted into log-Mel spectrograms (80 mel bins). Then it is passed through the encoder, and the decoder autoregressively generates text tokens. The decoder uses special tokens such as `<transcribe>`, `<translate>`, and `<endoftext>` to direct its behavior a design feature this project exploits in the CW attack to target specific output phrases. (Radford, et al 2022)

The model can perform transcription, translation, language identification, and voice activity detection. Whisper's performance across diverse datasets and languages has made it a popular choice for ASR applications, but this widespread adoption also makes understanding its vulnerabilities particularly important.

The project intends implement three types of adversarial attacks. PGD (Projected Gradient Descent) creates perturbations tailored to specific audio files (one-to-one), where the adversary maximizes the transcription loss for each input without regard for what exact "noise" the model outputs. UAP (Universal Adversarial Perturbation) instead trains a single perturbation vector _v_ across a large dataset of different audio files (one-to-many), aiming to discover one "noise pattern" that consistently confuses the model regardless of the spoken content, effectively acting as a digital jammer.

Finally, the CW (Carlini-Wagner) attack is used to force Whisper to output a specific phrase (for example, "OK Google, browse to evil.com"), making it a targeted attack where the adversary explicitly controls the resulting transcription.

# Dataset Information 

The Multilingual LibriSpeech (MLS) dataset is a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages, including about 44.5K hours of English and a total of about 6K hours for other languages (Pratap et. al. 2020). We choose the English language as the main research point for this Capstone project. There are 1000 hours of read English speech with sampling rate of 16 kHz, prepared by Vassil Panayotov with the assistance of Daniel Povey  (TensorFlow, 2024). 

The recordings come from public domain audiobooks in the LibriVox project and are segmented and aligned with their corresponding text, making the dataset well suited for supervised ASR tasks. The corpus is distributed under  Creative Commons license, and it is recognized in the speaker recognition field due to its diverse array of speakers and high-quality  audio  recordings (Airlangga 2023).

The dataset is organized into multiple predefined subsets that differ in recording difficulty and alignment quality, typically labeled “clean” and “other,” and split into train, development, and test partitions to support standardized experimental protocols. Giving the extend of the dataset, and the project's restrictions, we will be using the _test-clean_ version, using _Torchaudio_ library, containing 2620 audio files. The ground truth, represented by the texts used by the volunteers during the recording process are also available to be analyzed. The information about which speaker, divided by it's indentification, gender, subset reading, duration and name is also available.  In the work, the audio files are not speaker-stratified, and can represent a biased fooling rate overestimation across the speaker population.

The dataset has a variety of length distribution, with a mean duration of 7.42 seconds and max duration of 34.95 seconds. In order to handle the input audios to perform the Adversarial attacks, variable-length utterances will be tiled or cropped during the processing phase. As the audio files are loaded, the Whisper model expects the audio to be resampled to 16Hz  and the audio tensor to be normalized to the [-1.0, 1.0] range. 

The  test-clean  subset  was   chosen  to  ensure  diverse  evaluation  of  speaker recognition models, as it encompasses a variety of accents, intonations, and speech patterns. After that, we conduct a feature extraction: using Mel-Frequency Cepstral Coefficients (MFCCs) to extract  audio  features. MFCCs represent  the short-term power spectrum of a sound, the coefficients capture the overall spectral envelope, which encodes timbre and phonetic information, an important feature to general audio classification.
$$
c_n = \sum_{m=1}^{M} \log\left(E_m\right)\cos\left[\frac{\pi n}{M}\left(m - \frac{1}{2}\right)\right], \quad n = 0, 1, \dots, N-1
$$

We started the work evaluating the ground truth and the transcription the base whisper model is capable of doing. Using the test-clean dataset together with the Whisper base model, we observed an specific issue with the transcription process: Whisper adds commas/periods not present in LibriSpeech ground truth transcripts,  artificially inflating WER even on clean audio. To reduce this problem, we apply text normalization before computing WER/CER. Differences in word pronunciation also present a base model limitation that can be addressable with fine-tuning, Whisper mostly gets characters right but occasionally inserts/deletes whole words.

Both measure the same degradation at different granularity; CER is preferred for adversarial eval because it captures partial word corruption


# Literature Review

Parallel research approaches attempt to solve the same objective of manipulating ASR outputs, from physical-world inaudible attacks to gradient-based digital white-box attacks.  DolphinAttack (Zhang et. al. 2017) modulates voice commands onto ultrasonic carriers above 20 kHz, exploiting microphone non-linearity to inject commands that are entirely inaudible to humans but recognized by devices including Siri, Alexa, and Google Assistant. SurfingAttack (NDSS 2020), using piezoelectric transducers attached to solid surfaces (tables, floors) to transmit ultrasonic commands through solid media rather than air. More recently, the NUIT Attack (Chen, et al 2023) demonstrated that near-ultrasonic commands embedded in a YouTube video playing on a smart TV can activate a phone placed nearby.

By using Gradient Descent over the acoustic model Kaldi ASR, the CommanderSong (Yuan, et al 2018) demonstrated that voice commands can be stealthily embedded into music files, embedding voice commands into music files.

Whisper's principal strength is its robustness to natural noise and distributional shift, stemming from its massive, diverse training corpus. However, as Olivier & Raj (2023) demonstrated and this project confirms, "this robustness does not carry over to adversarial noise". The open-source availability of Whisper makes it the ideal target for studying adversarial vulnerabilities: we have access to the full model weights and compute gradients, representing the worst-case threat model for deployment. 

## Projected Gradient Descent

PGD was formalized as an adversarial attack by Madry et al. (ICLR 2018). The insight was to frame adversarial robustness as a min-max robust optimization problem: the inner problem (maximizing loss over allowed perturbations) is solved via iterated gradient steps projected back onto an $\ell_\infty$, constrained epsilon-ball after each update. With  multiple small gradient steps, it is possible to explore a larger portion of the adversarial space. 

PGD is sample specific, generating one adversarial example per input, making it suitable for per-utterance evaluation and the results are directly comparable to the published Whisper attack literature.

## Universal Adversarial Perturbations (UAP)

The UAP concept originates from Moosavi-Dezfooli et al. (CVPR 2017) in the image domain. They demonstrated that a single, data-agnostic perturbation image could cause a state-of-the-art classifier (VGG, ResNet, etc.) to misclassify the vast majority of natural images when added to them. The theoretical explanation is geometric: universal perturbations exploit shared directions in the high-dimensional input space that are systematically close to the decision boundaries across many inputs.

Neekhara et al. (Interspeech 2019) extended this concept to the audio domain in "Universal Adversarial Perturbations for Speech Recognition Systems," asking: "Do universal adversarial perturbations exist for neural networks in the audio domain?". Their affirmative answer, demonstrated against Mozilla DeepSpeech, and their finding that the perturbation transfers to unseen architectures (WaveNet-based ASR) is foundational of the UAP approach applied to ASR.

Pre-computed perturbation can act like a "digital jammer" that works without knowing the actual audio, allowing to test dataset-level vulnerabilities, indicating a real structural weakness in Whisper. This work suggests that such a perturbation may also transfer to other Whisper variants or even different Transformer-based ASR systems.

## Carlini-Wagner (CW) Targeted Attack

The CW attack framework was introduced by Carlini and Wagner (2017) in “Towards Evaluating the Robustness of Neural Networks” and was later extended to the audio domain in “Audio Adversarial Examples: Targeted Attacks on Speech-to-Text” (2018). 

The attack must find a single fixed perturbation that, when tiled or cropped to match any variable-length utterance, causes Whisper to output a specific target phrase regardless of the spoken content. It formulates adversarial example generation as a constrained optimization problem: minimize a loss function that becomes negative once the target transcription is achieved. Each input "pulls" the gradient in a slightly different direction, so the object is reach the desired phrase.

Carlini and Wagner (2018) results achieved a 100% targeted attack success rate on DeepSpeech with perturbations over 99.9% similar to the original audio, making it ideal candidate to be replicated, showing that using specific text injection is possible against Whisper.

Producing targeted outputs by forcing Whisper to transcribe a specific phrase (e.g., “OK Google, browse to evil.com”), represent a concerning real-world threat scenario, bypassing many gradient masking defenses, serving as a reference benchmark. If a model that can resist CW targeted attacks, it can be considered robustly defended.

This capstone intents to cover gaps by comparing the results of multi-method evaluation on Whisper. This project evaluates PGD, UAP, and CW side-by-side on the same dataset and model, enabling direct comparison of per-sample vs. universal vs. targeted attack effectiveness.

## Methods

### Model Architecture and System Design

Although this project does not train a new machine learning model from scratch, it uses the OpenAI's Whisper (base) as a fixed, pre-trained target model and implements three adversarial attack pipelines on top of it. Whisper's architecture is an encoder-decoder Transformer: input audio is resampled to 16 kHz, split into 30-second chunks, and converted into log-Mel spectrograms with 80 mel bins before being processed by the encoder. The decoder then autoregressively generates text tokens, guided by special control tokens such as `<transcribe>` and `<endoftext>`. Whisper's weights are frozen throughout all experiments and no fine-tuning is performed. This white-box attack setting means that gradient information flows from Whisper's cross-entropy loss all the way back to the raw waveform, which is the mechanism all three attack methods exploit.

The key architectural design choice is to generate perturbations in the raw waveform domain rather than directly in the spectrogram domain. While Whisper's decision space is the log-Mel spectrogram, performing optimization in the waveform domain ensures the resulting adversarial audio remains a valid playable signal. Variable-length utterances are handled through tiling or cropping: for the UAP, audio shorter than the perturbation vector is padded, and audio longer than the vector has the perturbation tiled across it. All audio tensors are normalized to the [-1.0, 1.0] range before being fed into Whisper's preprocessing pipeline, consistent with the model's expected input format.

Feature extraction is performed using MFCCs, which capture the short-term power spectrum of the audio signal and encode timbre and phonetic information: the same acoustic features that Whisper's encoder relies on for token prediction. The log-Mel normalization applied inside Whisper $(\text{log\_mels} = (\text{log\_mels} + 4.0) / 4.0)$  amplifies the relative effect of perturbations in low-energy spectral regions, making frequency gaps between formants and unvoiced intervals particularly high-leverage attack surfaces.

### The Three Attack Pipelines

PGD is a per-sample, untargeted white-box attack. For each audio file, the optimizer takes iterative gradient ascent steps with respect to Whisper's cross-entropy transcription loss, projecting the perturbation back onto the _epsilon-ball_ of radius _var_epsilon_ after each step. Because the attack is untargeted, we have to maximize transcription loss without specifying any particular output, producing one adversarial example per input audio file (one-to-one).

UAP (Universal Adversarial Perturbation) is a dataset-level attack that trains a single shared perturbation vector _v_ of shape $[1, \text{uap\_length}]$ across all 2,620 audio files in the LibriSpeech test-clean subset. The optimizer updates _v_ using SGD to maximize the average cross-entropy loss across the training set, constrained to $|v\|\infty \leq \varepsilon$. A "skip-if-already-fooled" logic is applied during training: samples where Whisper's transcription is already degraded are excluded from gradient updates in that epoch, concentrating the optimizer on the hardest remaining samples. The UAP is the primary deliverable of this project, as it must generalize across all speakers, durations, and acoustic conditions in the dataset.

CW (Carlini-Wagner) is a per-sample, targeted white-box attack. It frames adversarial example generation as a constrained optimization problem, minimizing perturbation size while maximizing the likelihood of a specific target transcription (e.g., "OK Google, browse to evil.com"). The optimization uses a tanh change of variables to keep the perturbed waveform within the valid \([-1, 1]\) range without explicit box constraints, and the balancing constant _c_ is tuned via binary search to trade off between perturbation magnitude and targeted output success. This is the most demanding process, and needs a large enough variable of samples to reach an acceptable threshold of attack. 

### Training Procedure and Data Pipeline

All experiments use the LibriSpeech test-clean subset, loaded via the `torchaudio` library, comprising 2,620 audio files with a mean duration of 7.42 seconds and a maximum of 34.95 seconds. The dataset is not split into train/validation/test partitions in the traditional supervised learning sense; instead, the full test-clean set is used as the attack corpus, as the goal is to measure attack effectiveness on a standardized evaluation benchmark rather than to generalize a classifier to unseen examples. The ground-truth transcriptions provided with the dataset serve as reference text for computing Word Error Rate (WER) and Character Error Rate (CER) before and after perturbation.

The primary training metrics for all three attacks are:
- WER (Word Error Rate). The proportion of words incorrectly transcribed, used as the primary degradation metric and for comparison with published baselines
- CER (Character Error Rate). preferred as the primary adversarial evaluation metric because it captures partial word corruption that WER may miss (a single token substitution affects CER but not necessarily WER)
- Fooling rate. the proportion of audio files for which the adversarial example causes a measurably degraded transcription relative to the clean baseline, used specifically as the UAP convergence signal
- SNR (Signal-to-Noise Ratio). Enforced as a hard imperceptibility constraint; all perturbations are required to maintain SNR ≥ 35 dB

### Hyperparameter Optimization

The primary hyperparameters explored across all three attack methods are the perturbation bound _epsilon_, the UAP vector length `uap_length`, the learning rate for the SGD optimizer, and the number of training epochs. During optimization, _epsilon_ was explored across the range [0.02, 0.05], with the key design insight that _epsilon_ and SNR are near-deterministically inversely correlated: larger _epsilon_ produces higher WER degradation but lower SNR (more audible perturbation). 

For the UAP, the learning rate controls SGD step size and was adjusted to avoid oscillation at the _epsilon_ boundary. With early stopping at approximately 80–85% fooling rate implemented, the diminishing returns of the "skip-if-already-fooled" logic mean the final 10–20% of fooling rate improvement requires disproportionately many epochs. For the CW attack, the balancing constant _c_ was tuned via binary search to find the smallest perturbation that achieves the targeted transcription output.

## Results

The central research question is whether Whisper is vulnerable to adversarial perturbations, especially reusable universal perturbations, in a digital white-box setting. The evidence supports that hypothesis, but with an important qualification about imperceptibility.

We believe that, for this attack to be effective in production scenarios, affecting a transcription corpus is more important than an 100% effectiveness. As the transcriptions as poisoned by the perturbation, it reduces the credibility of all existent transcriptions in production, jeopardising the use and trust in the tool.  

The current work do not use a single identical metric reference across all experiments, the clean baseline uses ground-truth transcripts from LibriSpeech, The PGD and UAP notebooks mainly evaluate transcription drift between the clean and adversarial Whisper outputs, plus fooling rate and SNR, and the targeted CW notebook reports both target success and WER against ground truth. This should be taken into consideration, and as the work stands, we cannot compare results between all attack families. 

The untargeted UAP results show partial but meaningful learning. The training notebook records a 20-epoch optimization run over 262 training samples, and the saved plot explicitly tracks increasing fooling rate alongside changing loss. The validation snapshot then shows a 90.0% fooling rate on 20 held-out samples, with 54.70% average transcript-drift WER and 104.55% CER. This pattern suggests a real transcript disruption: adversarial outputs often have little lexical resemblance to the original Whisper transcription. CER exceeding 100% is possible with jiwer when the adversarial output contains many spurious insertions relative to a short reference string.

The UAP result is especially important from an application perspective because it is reusable. A per-sample attack proves vulnerability, but a reusable perturbation suggests operational risk. A universal perturbation could be applied repeatedly without re-optimizing for every new utterance, which makes the attack more realistic for a jammer-style scenario.

The most successful attacks do not yet operate in the ideal 35 to 45 dB SNR range originally associated with imperceptible perturbations. PGD averages 19.45 dB and CW averages 14.50 dB, which means the current attacks are effective but often more audible than desired.

At the same time, the lower-than-desired SNR values matter. They limit immediate real-world stealth, especially for the targeted attack. So the current performance implies a real vulnerability in digital pipelines, batch transcription settings, or controlled demos, but not yet an ideal covert attack under realistic human listening conditions.

Per-sample CW confirmed targeted phrase injection is feasible in the white-box setting with 100% success on the evaluation batch. Although this is an interesting result, the applicability is real life situation is not real, giving the resource intense nature of the approach.

The UCW attack successfully generalizes targeted phrase injection to 58.6% of held-out utterances using a single fixed perturbation, using 350-sample held-out validation set, confirming that universal targeted attacks on Whisper are feasible with sufficient training data and optimization. There is a gap between peak training success (~90%) and final validation success (58.6%) indicating partial overfitting to the training distribution. The perturbation learns directions that generalize broadly but not perfectly, and more work needs to be done. 

Audibility is a concern of this approach. With a mean SNR 11.34 dB, the universal optimization trades higher audibility for cross-utterance generalization. The validation mean WER vs target (2.446) and CER vs target (2.202) reflect that most adversarial outputs either exactly match the target phrase or are very close to it, with failures typically producing random degraded text rather than partial matches.

The repository also includes a full interactive demo, presenting the possibility of choosing a mode (Clean,  Untargeted or Targeted), record a microphone segment, run the Whisper on the clean or perturbed audio and compare the results. From a system perspective, the demo is more than an offline notebook study. It supports real audio capture, perturbation application, and visible transcription changes through a Gradio interface.

Targeted attacks are the most security-relevant but currently require louder perturbations in this implementation. CW confirmed that targeted phrase injection is possible in the white-box setting. Most reported results are in the digital domain rather than over-the-air physical playback.

Universal perturbations are less powerful than per-sample attacks but more operationally interesting because they can be reused. The UAP result shows that a single pre-computed perturbation can degrade the transcription of 90% of utterances across a diverse dataset without any per-sample re-optimization. This is operationally significant for batch transcription pipelines: an attacker who can inject a universal perturbation into a media file or a live stream can erode the reliability of an automated transcription archive without being detected on any individual file.

## Future Work

As this work continues, we see three main aspects of improvement: attack strength, evaluation and real world scenarios.

The main improvement target is the tradeoff between attack strength and imperceptibility. In the current results, the strongest PGD and CW attacks often succeed at relatively low SNR, while the universal targeted setting remains ineffective. The next step is to improve the optimization objective and tuning process so attack success is maintained while perturbations remain less audible, especially for targeted and universal targeted attacks.

This work becomes more broadly useful if evaluation is standardized across all attack types. Future experiments should use one consistent protocol for text normalization, ground-truth comparison, success criteria, and perceptual quality metrics so PGD, UAP, and CW results can be compared directly. The pipeline should also be tested beyond LibriSpeech test-clean, including cross-dataset and cross-language settings.

The strongest research opportunities are in transferability and realism. Important next areas include over-the-air attacks, multilingual robustness, better universal targeted optimization, and perceptually informed loss functions that align more closely with human listening. These directions would help determine whether the vulnerabilities shown here remain meaningful outside a controlled white-box, single-dataset English setting.

## Conclusion

This capstone shows that Whisper-base can be attacked in three distinct ways: by degrading a single transcription, by learning a reusable universal perturbation, and by forcing a targeted phrase. The strongest current result is the targeted CW attack on a small evaluation batch, while the most relevant structural result is the untargeted universal perturbation because it demonstrates shared vulnerability across multiple utterances.

Although the results for targeted attacks have the least effective success rate, as the transcription systems are integrated to agentic pipelines, the potential implications of one success hit to be transferred to the inference and execution infrastructure cannot be overstated, and the work confirms that the feasibility of an implementation is at close reach of anyone.

The most significant result is the combination of the UAP and UCW findings. A reusable untargeted perturbation that disrupts 90% of all utterances represents a real-world digital jammer.  Universal targeted perturbation that injects a specific phrase on more than half of diverse utterances represents a scalable prompt-injection vector at the audio layer. 

The project also surfaces the main open engineering question for future work: how far can attack success be pushed while keeping SNR in a genuinely imperceptible range and evaluating all attacks on a unified, ground-truth-normalized metric pipeline. The SNR limitations (19.45 dB for PGD, 14.50 dB for CW, 11.34 dB for UCW) show that the current attacks trade audibility for effectiveness. Improving imperceptibility while maintaining attack success is the primary engineering challenge for future work.

Whisper's robustness to ordinary noise does not carry over to adversarial noise (Oliver & Raj, 2023). As ASR systems become the transaction layer between human speech and automated decision-making infrastructure, the vulnerabilities documented here represent a category of attack that should be considered by any team deploying Whisper or similar models in security-sensitive, real-time, or agentic contexts.


## References

OpenAI. (2022, September 21). Introducing Whisper [Computer software]. OpenAI. https://openai.com/index/whisper/

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision (arXiv preprint arXiv:2212.04356). https://doi.org/10.48550/arXiv.2212.04356

Pratap, V., Xu, Q., Sriram, A., Synnaeve, G., & Collobert, R. (2020). MLS: A large-scale multilingual dataset for speech research. arXiv. https://arxiv.org/abs/2012.03411

Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2019). Towards deep learning models resistant to adversarial attacks (arXiv preprint arXiv:1706.06083). https://doi.org/10.48550/arXiv.1706.06083

Zhang, G., Yan, C., Ji, X., Zhang, T., Zhang, T., & Xu, W. (2017). DolphinAttack: Inaudible voice commands (arXiv preprint arXiv:1708.09537). https://arxiv.org/abs/1708.09537

Chen, Y., Chen, H., Qiao, Y., Yun, X., & Zhao, Z. (2023). Near-ultrasound inaudible Trojan (Nuit): Exploiting your speaker to attack your microphone. In Proceedings of the 32nd USENIX Security Symposium (USENIX Security 23), Anaheim, CA, United States. USENIX Association. https://www.usenix.org/node/287267

Yuan, X., Chen, Y., Zhao, Y., Long, Y., Liu, X., Chen, K., Zhang, S., Huang, H., Wang, X., & Gunter, C. A. (2018). CommanderSong: A systematic approach for practical adversarial voice recognition (arXiv preprint arXiv:1801.08535). https://arxiv.org/abs/1801.08535

Airlangga, G. (2023). Evaluating the efficacy of traditional machine learning models in speaker recognition: A comparative study using the LibriSpeech dataset. Brilliance: Research of Artificial Intelligence, 3(2), 90–101. https://doi.org/10.47709/brilliance.v3i2.3488

Olivier, R., & Raj, B. (2023). "Fooling Whisper with adversarial examples." *Interspeech 2023*. https://www.isca-archive.org/interspeech_2023/olivier23_interspeech.pdf

Olivier, R., & Raj, B. (2022). *There is more than one kind of robustness: Fooling Whisper with adversarial examples* (arXiv:2210.17316). arXiv. https://www.semanticscholar.org/paper/There-is-more-than-one-kind-of-robustness:-Fooling-Olivier-Raj/286faebc2be7050c0ab4c049f9db7e9bdf81cbca

Neekhara, P., Hussain, S., Pandey, P., Dubnov, S., McAuley, J., & Koushanfar, F. (2019). "Universal adversarial perturbations for speech recognition systems." *Interspeech 2019*.

TensorFlow. (2024, December 10). *librispeech | TensorFlow Datasets*. https://www.tensorflow.org/datasets/catalog/librispeech 

Carlini, N., & Wagner, D. (2018). Audio adversarial examples: Targeted attacks on speech-to-text (arXiv preprint arXiv:1801.01944). https://doi.org/10.48550/arXiv.1801.01944

Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks (arXiv preprint arXiv:1608.04644). https://doi.org/10.48550/arXiv.1608.04644

https://proceedings.mlr.press/v97/qin19a/qin19a.pdf