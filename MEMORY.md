# Codebase Architectural Memory & Observations

## Codebase Structure
- `model.py`: PyTorch CNN network definition (`convNet`), data generator (`train_data_builder`, `infer_data_builder`), training loop, and inference loop.
  - Architecture: 3 input channels (from 3 different FFT window sizes: 1024, 2048, 4096).
  - Mel frequency bands: 80 bands. Temporal sound snippet size: 15 frames.
  - Dropout: 0.5 across dense layers (`1120 -> 256 -> 120 -> 1`).
  - Sigmoid output produces onset probability $[0, 1]$.
- `music_processor.py`:
  - `Audio` class: reads `.wav`/`.ogg` audio files using SoundFile, parses `.tja` (Taiko no Tatsujin format) chart files with BPM and measure changes.
  - `fft_and_melscale`: Computes multi-scale log-mel spectrograms using Blackman-Harris windows and Librosa filterbanks.
- `infer.py`: Runs forward inference over preprocessed song features (`test_data.pickle`) and saves prediction vectors (`don_inference.pickle`, `ka_inference.pickle`).
- `synthesize.py`:
  - Performs peak picking using Librosa's `peak_pick` algorithm.
  - Synthesizes detected onsets with original don/ka sample audio into `inferred_music.wav`.
  - Exports detected onsets into standard Taikosanjiro `.tja` chart format.
- `dashboard.py`: Unified single-page FastAPI web application providing end-to-end access to preprocessing, training, inference, peak-picking, chart generation, live microphone streaming, interactive waveform scrubber, and Mel spectrogram heatmap visualizer.
- `tray_app.py`: System tray companion for desktop server lifecycle control.
- `chart_exporters.py`: Multi-format exporters for osu! Taiko, StepMania, and Beat Saber.
- `export_onnx.py`: ONNX graph exporter and ONNX Runtime inference engine.
- `audio_stream.py`: Sliding buffer audio stream generator and real-time ring buffer onset detector.

## Key Discovered Constraints
1. Modern Librosa (>=0.10.0) mandates keyword-only arguments for `librosa.util.peak_pick` and `librosa.filters.mel`.
2. SciPy (>=1.14.0) moved windowing functions to `scipy.signal.windows`.
3. Numba cannot compile arbitrary Python objects in nopython mode without strict jitclass definitions; standard Python iteration over the list of audio objects is faster and avoids compilation failure.
