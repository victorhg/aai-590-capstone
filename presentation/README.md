# Presentation Materials

This folder contains the documentation and scripts for the final presentation of the "Audio Adversarial Attacks on Whisper" capstone project.

## Files Included:
1.  **presentation_guide.md**: Structured guide with timing, slide outline, demo instructions, and results tables.
2.  **teleprompter_script.md**: Word-for-word script (~12 minutes) with stage directions, pauses, and demo cues. Read from this during the presentation.
3.  **Q&A_common.md**: Anticipated questions and answers with actual project metrics.
4.  **README.md**: This file.

## Before the Presentation:
1.  **Launch the demo**: Open `notebooks/08_train_demo_targeted_attack.ipynb` and run all cells to start the Gradio interface.
2.  **Verify perturbation files**: Ensure `results/universal_perturbation_v.pt` and `results/ucw_delta.pt` are present.
3.  **Test all three modes**: Clean, Untargeted UAP, and Targeted CW Injection — both single-audio and streaming.
4.  **Prepare slides**: Use the slide outline in `presentation_guide.md` to build slides. Key visuals: Whisper architecture diagram, attack pipeline, results comparison table.

## Demo Flow:
1.  **Clean Mode** — establish baseline (WER ≈ 4.78%).
2.  **Untargeted UAP** — show transcript corruption.
3.  **Targeted CW Injection** — show phrase injection ("access evil website").
4.  For streaming, speak for at least 3–4 seconds in targeted mode to allow sufficient audio accumulation.

## Key Numbers to Remember:
- PGD: 92% success, 26.58 dB SNR
- UAP: 60% fooling rate
- UCW: 58.6% success on 350 validation samples, 11.34 dB SNR
- Baseline: 4.78% WER on clean audio

