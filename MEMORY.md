# Codebase Architectural Memory & Observations

## Codebase Structure
- `chart_exporters.py`: Multi-format exporters for osu! Taiko, StepMania, Beat Saber, Standard MIDI Files (SMF Type 0), and beat grid quantizer `quantize_timestamps_to_grid()`.
  - `export_midi(timestamps_don, timestamps_ka, output_path, bpm, track_name)`: Zero-dependency pure-Python binary SMF Type 0 generator with variable-length quantity (VLQ) delta-time encoding, microsecond tempo events, and General MIDI Percussion Channel 10 mappings (Don = 36 Bass Drum 1, Ka = 38 Acoustic Snare).
- `model_crnn.py`: Hybrid Convolutional Recurrent Neural Network with temporal Multihead Rhythm Attention (`CRNNOnsetMaxNet`, `RhythmAttention`).
  - Architecture: 3 input channels from 3 FFT windows (1024, 2048, 4096).
  - Mel frequency bands: 80 bands. Temporal sound snippet size: 15 frames.
  - Conv2D feature extraction followed by adaptive frequency max-pooling to collapse spectral dimensions into temporal feature embeddings.
  - Bidirectional GRU (hidden_size=64, 2 layers) sequence modeling across time frames.
  - Multihead Rhythm Attention (4 heads, d_model=128) with residual connection and LayerNorm.
  - Sigmoid output produces onset probability $[0, 1]$.
- `model.py`: Original PyTorch CNN network definition (`convNet`), data generator (`train_data_builder`, `infer_data_builder`), training loop, and inference loop.
  - Architecture: 3 input channels, 80 Mel frequency bands, 15 frames.
  - Dropout: 0.5 across dense layers (`1120 -> 256 -> 120 -> 1`).
  - Sigmoid output produces onset probability $[0, 1]$.
- `music_processor.py`:
  - `Audio` class: reads `.wav`/`.ogg` audio files using SoundFile, parses `.tja` (Taiko no Tatsujin format) chart files with BPM and measure changes.
  - `fft_and_melscale`: Computes multi-scale log-mel spectrograms using Blackman-Harris windows and Librosa filterbanks.
- `infer.py`: Runs forward inference over preprocessed song features (`test_data.pickle`) with configurable architecture (`--arch {cnn,crnn}`) and saves prediction vectors (`don_inference.pickle`, `ka_inference.pickle`).
- `synthesize.py`:
  - Performs peak picking using Librosa's `peak_pick` algorithm.
  - Synthesizes detected onsets with original don/ka sample audio into `inferred_music.wav`.
  - Exports detected onsets into standard Taikosanjiro `.tja`, osu! Taiko `.osu`, StepMania `.sm`, Beat Saber `.dat`, and MIDI `.mid` chart formats.
- `dashboard.py`: Unified single-page FastAPI web application providing end-to-end access to preprocessing, training, inference (CNN vs. CRNN), peak-picking, chart generation, live microphone streaming, interactive waveform scrubber, Mel spectrogram heatmap visualizer, tap tempo / beat grid quantizer, and multi-format chart downloads.
- `tray_app.py`: System tray companion for desktop server lifecycle control.
- `export_onnx.py`: ONNX graph exporter for both CNN and CRNN models and ONNX Runtime inference engine.
- `audio_stream.py`: Sliding buffer audio stream generator and real-time ring buffer onset detector.

## Key Discovered Constraints
1. Modern Librosa (>=0.10.0) mandates keyword-only arguments for `librosa.util.peak_pick` and `librosa.filters.mel`.
2. SciPy (>=1.14.0) moved windowing functions to `scipy.signal.windows`.
3. Numba cannot compile arbitrary Python objects in nopython mode without strict jitclass definitions; standard Python iteration over the list of audio objects is faster and avoids compilation failure.
4. PyTorch 2.x `torch.onnx.export` with recurrent networks (GRU) and multihead attention requires `dynamo=False` (legacy TorchScript) and explicit `dynamic_axes` to avoid batch dimension freezing.
5. Standard MIDI Files (SMF) use big-endian integer headers (`>I`, `>H`) and variable-length quantity (VLQ) delta times (7 bits per byte, MSB set on all except the final byte).
