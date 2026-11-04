# VISION: ODCNN Musical Onset Detection & Rhythm Game AI Engine

## 1. Core Foundational Concepts
ODCNN is an advanced neural onset detection and automated chart generation system for rhythm games (such as Taiko no Tatsujin / Taikosanjiro). Based on Jan Schlüter and Sebastian Böck's research ("Improved Musical Onset Detection with Convolutional Neural Networks"), the project transforms audio time-series signals into multiscale log-mel spectrogram representations and leverages convolutional neural architectures to isolate harmonic and percussive transients.

## 2. Ultimate Goal
To deliver an autonomous, end-to-end studio where users can:
1. Ingest any musical piece in various audio formats.
2. Automatically extract features across multi-resolution FFT windows.
3. Train, fine-tune, or use pre-trained dual CNN models targeting distinct note timbres (`don` - bass drum / percussive, `ka` - rimshot / harmonic).
4. Run high-throughput batched inference on audio tracks.
5. Apply adaptive peak-picking algorithms with adjustable thresholds, refractory periods, and spectral smoothing.
6. Synthesize the detected onsets into Taiko chart formats (`.tja`) and reconstructed audio playback files (`.wav`).
7. Manage everything via a unified, single-page, real-time web dashboard paired with a system tray controller.

## 3. User-Satisfaction Design
- **Single Page Simplicity:** No fragmentation across disjointed pages. High-value workflows (inference, synthesis, playback) are positioned prominently at the top, supported by comprehensive data prep and model management below.
- **Immediate Sensory Feedback:** Integrated interactive HTML5 audio players for both raw audio and synthesized onset tracks.
- **Explainability & Tooltips:** Every hyperparameter, parameter slider, and file option includes descriptive tooltips explaining the underlying signal processing or neural network concept.
- **Graceful Execution & Observability:** Live server resource monitoring (CPU, RAM, GPU availability), real-time process execution logs, and clear visual indicators.
