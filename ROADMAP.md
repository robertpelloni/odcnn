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
- [ ] Extend CNN architecture to modern Conformer or CRNN (Convolutional Recurrent Neural Network) backbones.
- [ ] Add roll/slider (dendera/balloon) detection using sustained spectral energy analysis.

### Milestone 3: Portable Runtime & Microcontroller Integration (v1.3.0 - Completed)
- [x] Export PyTorch CNN to standardized ONNX graphs (`export_onnx.py`).
- [x] Support ONNX Runtime engine execution directly inside the dashboard studio.

### Milestone 4: Streaming Audio Pipeline & Custom Dataset Ingestion (v1.4.0 - Completed)
- [x] Live audio sliding buffer streaming simulator (`audio_stream.py`).
- [x] Real-time onset detection using circular ring buffers, short-term RMS ratios, and spectral centroid classification.
- [x] Custom user music upload endpoint (`POST /api/upload_audio`) and real-time streaming simulation UI card.

### Milestone 5: Unified Automation & CLI Orchestration (v1.5.0 - Current)
- [x] Consolidated CLI entry point (`run.py`) unifying `studio`, `train`, `infer`, `synthesize`, `export-onnx`, and `stream-sim`.
- [x] Comprehensive CLI unit test verification (`tests/test_cli.py`).
- [ ] WebRTC / WebSocket streaming for sub-5ms low latency live line-in detection.
- [ ] Low-latency C++/Rust ONNX Runtime standalone runner for embedded rhythm hardware.
- [ ] WebAssembly / WebAudio client-side onset inference in browser.
