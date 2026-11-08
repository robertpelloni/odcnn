# ROADMAP: Long-Term Architectural Milestones

### Milestone 1: Core Engine Modernization & Unified Dashboard (v1.0.0 - Completed)
- [x] Modernize PyTorch and Librosa APIs (resolve deprecations in peak-picking and Mel-spectrogram computation).
- [x] Repair SciPy window functions for current SciPy versions.
- [x] Build unified, single-page, responsive web dashboard with dark-mode Taiko styling.
- [x] Implement real-time process management and live server logging.
- [x] Implement system tray controller with server start/stop and quit capabilities.
- [x] Implement automated end-to-end testing and frontend verification.

### Milestone 2: Multi-Model Architecture & General Rhythm Game Support (v1.2.0 - Current)
- [x] Support custom game chart formats: StepMania (.sm), osu! (.osu) Mode 1 Taiko.
- [x] Support VR game chart formats: Beat Saber (.dat v2 schema).
- [ ] Extend CNN architecture to modern Conformer or CRNN (Convolutional Recurrent Neural Network) backbones.
- [ ] Add roll/slider (dendera/balloon) detection using sustained spectral energy analysis.

### Milestone 3: Real-Time Stream Ingestion & Microcontroller Integration (v1.3.0)
- [ ] Live microphone and line-in onset detection via PortAudio / PyAudio streams.
- [ ] Low-latency C++/Rust ONNX Runtime export for embedded rhythm hardware.
- [ ] WebAssembly / WebAudio client-side onset inference in browser.
