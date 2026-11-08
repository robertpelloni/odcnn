# ROADMAP: Long-Term Architectural Milestones

### Milestone 1: Core Engine Modernization & Unified Dashboard (v1.0.0 - Completed)
- [x] Modernize PyTorch and Librosa APIs (resolve deprecations in peak-picking and Mel-spectrogram computation).
- [x] Repair SciPy window functions for current SciPy versions.
- [x] Build unified, single-page, responsive web dashboard with dark-mode Taiko styling.
- [x] Implement real-time process management and live server logging.
- [x] Implement system tray controller with server start/stop and quit capabilities.
- [x] Implement automated end-to-end testing and frontend verification.

### Milestone 2: Multi-Model Architecture & General Rhythm Game Support (v1.2.0 - Completed)
- [x] Support custom game chart formats: StepMania (.sm), osu! (.osu) Mode 1 Taiko.
- [x] Support VR game chart formats: Beat Saber (.dat v2 schema).
- [ ] Add roll/slider (dendera/balloon) detection using sustained spectral energy analysis.

### Milestone 3: Portable Runtime & Microcontroller Integration (v1.3.0 - Completed)
- [x] Export PyTorch CNN to standardized ONNX graphs (`export_onnx.py`).
- [x] Support ONNX Runtime engine execution directly inside the dashboard studio.

### Milestone 4: Streaming Audio Pipeline & Custom Dataset Ingestion (v1.4.0 - Completed)
- [x] Live audio sliding buffer streaming simulator (`audio_stream.py`).
- [x] Real-time onset detection using circular ring buffers, short-term RMS ratios, and spectral centroid classification.
- [x] Custom user music upload endpoint (`POST /api/upload_audio`) and real-time streaming simulation UI card.

### Milestone 5: Unified Automation & CLI Orchestration (v1.5.0 - Completed)
- [x] Consolidated CLI entry point (`run.py`) unifying `studio`, `train`, `infer`, `synthesize`, `export-onnx`, and `stream-sim`.
- [x] Comprehensive CLI unit test verification (`tests/test_cli.py`).

### Milestone 6: Live WebSocket Microphone Streaming (v1.6.0 - Completed)
- [x] Bidirectional WebSocket audio streaming endpoint in FastAPI (`/ws/stream`).
- [x] Client-side microphone capture (WebAudio API `ScriptProcessorNode` to Float32 PCM) directly piped to server.
- [x] Real-time visual hit indicators (animated Red Don and Blue Ka flash badges) in the Unified Studio.
- [x] End-to-end WebSocket automated test suite (`tests/test_pipeline.py`) and Playwright validation.

### Milestone 7: Interactive Audio Waveform Visualizer & Rhythm Onset Scrubber (v1.7.0 - Completed)
- [x] Downsampled audio waveform envelope generation API (`/api/audio_waveform`).
- [x] HTML5 Canvas rendering of dual min/max waveform with color-coded rhythm onset markers (Red Don / Blue Ka).
- [x] Interactive audio scrubbing and timeline synchronization with studio audio playback element.
- [x] Waveform endpoint unit tests and Playwright interaction verification.

### Milestone 8: Mel-Frequency Spectrogram Heatmap Visualizer (v1.8.0 - Completed)
- [x] Mel-frequency spectrogram data generation API (`/api/spectrogram_data`) with 48-mel frequency bins and downsampled frame matrix.
- [x] High-performance HTML5 Canvas spectral heatmap visualizer with dynamic Magma/Inferno color ramp.
- [x] Dual synchronized timeline scrubbing across waveform, spectrogram, and audio playback.
- [x] Automated unit testing and Playwright browser visual verification.

### Milestone 9: Advanced Neural Architectures (v2.0.0 - Completed)
- [x] Convolutional Recurrent Neural Network with temporal Multihead Rhythm Attention (`CRNNOnsetMaxNet` in `model_crnn.py`).
- [x] Bi-directional GRU recurrent layer for context-aware sequence modeling across onset frames.
- [x] Multihead temporal attention mechanism weighting rhythmic anticipation.
- [x] Dual model architecture support (`cnn` vs. `crnn`) across pipeline (`infer.py`, `dashboard.py`, `run.py`, `export_onnx.py`).
- [x] Dynamic batch axis ONNX export and ONNX Runtime execution for CRNN architectures.

### Milestone 10: Beat Grid Snapping & Rhythm Chart Quantization (v2.1.0 - Completed)
- [x] BPM Tap Tempo real-time calculator with interval smoothing and pause reset.
- [x] Mathematical beat grid snapping engine across standard musical subdivisions (1/4 to 1/32).
- [x] Multi-format chart re-quantization endpoint (`/api/chart/quantize`) rewriting `.tja`, `.osu`, `.sm`, and `.dat` files.
- [x] Interactive studio UI card for tap tempo, grid division selection, and one-click chart quantization.

### Milestone 11: Standard MIDI File Export & DAW Integration (v2.2.0 - Completed)
- [x] Pure-Python binary Standard MIDI File (SMF Type 0) exporter `export_midi(...)` with variable-length quantity delta encoding.
- [x] General MIDI percussion channel 10 mapping for Don (Kick 36) and Ka (Snare 38).
- [x] Synthesis pipeline generation (`inferred_notes.mid`) and `/api/download/midi` download endpoint in FastAPI Studio.
- [x] Studio UI "Download .MID" button and unit test / Playwright integration verification.

### Milestone 12: Embedded Integration & Audio Stems (Next)
- [ ] Dynamic multi-channel audio stems separation prior to onset feature extraction.
- [ ] Low-latency C++/Rust ONNX Runtime standalone runner for embedded rhythm hardware.
- [ ] WebAssembly / WebAudio client-side onset inference in browser without backend roundtrip.
