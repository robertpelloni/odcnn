# CHANGELOG

## [2.2.0] - 2025-11-08
### Added
- **Standard MIDI File (SMF Type 0) Exporter (`chart_exporters.py`):**
  - Designed and implemented a pure-Python binary SMF Type 0 exporter `export_midi(...)` with zero external dependencies.
  - Generates standard `MThd` and `MTrk` binary chunks with variable-length quantity (`VLQ`) delta-time encoding.
  - Maps rhythm onsets to General MIDI Percussion Channel 10 (`0x09`): Don maps to MIDI Pitch 36 (Bass Drum 1 / C2) and Ka maps to MIDI Pitch 38 (Acoustic Snare / D2).
  - Encodes microsecond tempo meta events (`0xFF 0x51 0x03`), track name meta events (`0xFF 0x03`), note-on events (`0x99`), note-off events (`0x89`), and End-of-Track meta events (`0xFF 0x2F 0x00`).
- **Pipeline & DAW Integration:**
  - Integrated `export_midi(...)` into `synthesize.py` generating `inferred_notes.mid` alongside `.tja`, `.osu`, `.sm`, and `.dat` chart files.
  - Integrated MIDI regeneration into the beat grid re-quantization endpoint `/api/chart/quantize` in `dashboard.py`.
- **FastAPI Studio REST API & UI Controls:**
  - Added `/api/download/midi` download endpoint in `dashboard.py` with `audio/midi` media type and attachment headers.
  - Added interactive "Download .MID" button (`#downloadMidiBtn`) with tooltips to the unified Studio Chart Exporters & Downloads card.
- **Automated Test Suite & Frontend Verification:**
  - Added unit test `test_export_midi_and_download_endpoint` in `tests/test_pipeline.py` verifying binary header markers (`MThd`, `MTrk`), channel assignments, event ordering, and HTTP download response.
  - Updated Playwright interaction test `tests/verify_frontend_interactions.py` verifying DOM presence and button interaction, validated with visual screenshot inspection.

## [2.1.0] - 2025-11-08
### Added
- **Interactive BPM Tap Tempo Calculator:**
  - Added real-time tap tempo estimator in `dashboard.py` calculating tempo from consecutive tap intervals with automatic pause resets (>2.5s) and running average smoothing.
  - Interactive "TAP TEMPO" UI button and live BPM readout badge displaying computed tempos in real time.
- **Beat Grid Quantization Engine (`chart_exporters.py`):**
  - Added `quantize_timestamps_to_grid(timestamps_sec, bpm, division, offset_sec)` implementing musical beat grid snapping across standard subdivisions (1/4 quarter notes, 1/8 eighth notes, 1/12 triplets, 1/16 sixteenth notes, 1/24 triplet sixteenths, 1/32 thirty-second notes) with duplicate deduplication.
- **Universal Multi-Format Re-Quantization Endpoint (`/api/chart/quantize`):**
  - Added REST endpoint in `dashboard.py` that parses existing `.tja` note lines, extracts Don and Ka onset timestamps, snaps them to the selected BPM grid, rewrites the `.tja` measure structure, and regenerates all exported formats (`.osu`, `.sm`, `.dat`) on the fly.
- **UI Integration in Unified Studio:**
  - Designed and added interactive card: "⏱️ BPM Tap Tempo & Beat Grid Quantizer" with BPM input, musical subdivision dropdown, beat offset control, and one-click "Quantize Charts to Grid" execution.
  - Real-time result badges and status feedback.
- **Comprehensive Unit & Frontend Verification:**
  - Added unit tests `test_quantize_timestamps_algorithm` and `test_quantize_charts_api_endpoint` in `tests/test_pipeline.py`.
  - Added automated Playwright interaction test verifying the Tap Tempo button, grid quantization API execution, and visual screenshot fidelity.

## [2.0.0] - 2025-11-08
### Added
- **CRNN & Multihead Rhythm Attention Neural Architecture (`model_crnn.py`):**
  - Designed and implemented `CRNNOnsetMaxNet`, a hybrid architecture coupling a 3-channel Conv2D feature extractor with a bidirectional GRU sequence model and temporal Multihead Rhythm Self-Attention (`RhythmAttention`).
  - Added temporal pooling layer that reduces 80 Mel frequency bands down to a compact temporal representation prior to bi-GRU sequence modeling.
  - Multihead attention layer dynamically weights onset anticipation across temporal sound context windows.
  - Provided automatic weight initialization helper `initialize_default_crnn_weights()` producing pre-trained default model checkpoints `models/don_crnn.pth` and `models/ka_crnn.pth`.
- **Full Inference & Pipeline Integration:**
  - Extended `infer.py` with `model_arch` parameter supporting both `cnn` and `crnn` backbones.
  - Updated CLI `run.py` to accept `--arch {cnn,crnn}` for onset inference.
  - Enhanced FastAPI studio backend (`dashboard.py`) with `arch` selection parameter in `/api/action/infer` and interactive select dropdown in the Studio UI (`#inferArch`).
- **CRNN ONNX Model Export & ONNX Runtime Support:**
  - Updated `export_onnx.py` to support exporting both CNN and CRNN architectures with dynamic batch axes (`--arch crnn`).
  - Compatible with `ONNXOnsetDetector` for multi-batch low-latency inference in ONNX Runtime.
- **Dedicated Test Suite & Playwright Verification:**
  - Added `tests/test_crnn.py` with 5 unit tests covering attention mechanism, forward pass output bounds, gradient backprop, song inference, and ONNX Runtime batch execution (expanding test suite to 21 tests, all passing).
  - Updated Playwright test `tests/verify_frontend_interactions.py` checking the model architecture selector, confirmed via visual UI snapshot.

## [1.8.0] - 2025-11-08
### Added
- **Mel-Frequency Spectrogram Heatmap Visualizer:**
  - Added `/api/spectrogram_data` endpoint in `dashboard.py` generating downsampled 48-mel frequency spectral energy matrix (20Hz - 11kHz, normalized 0.0 - 1.0) using cached STFT.
  - Implemented an HTML5 Canvas spectral heatmap visualizer in `dashboard.py` using a dynamic Magma/Inferno color ramp (dark purple -> magenta -> orange -> yellow -> white) with frequency axis tick markers and frequency band division indicator lines.
  - Synchronized spectrogram timeline scrubbing with the waveform canvas and HTML5 audio player.
- **Spectrogram Unit Testing:**
  - Added `test_spectrogram_data_endpoint` in `tests/test_pipeline.py` verifying downsampled mel matrix shape, frequency ranges, normalization, and note timestamp metadata.
- **Automated Frontend Verification:**
  - Updated Playwright test `tests/verify_frontend_interactions.py` checking `#spectrogramContainer`, `#spectrogramCanvas`, `#spectroStatusBadge`, and two-way scrubbing interaction.

## [1.7.0] - 2025-11-08
### Added
- **Interactive Audio Waveform Visualizer & Rhythm Onset Scrubber:**
  - Added `/api/audio_waveform` endpoint in `dashboard.py` generating downsampled dual min/max envelopes (1000 data points) and parsed rhythm note timestamps directly from generated `.tja` charts.
  - Implemented an HTML5 Canvas waveform visualizer featuring envelope drawing, overlaid rhythm onset hit markers (Red Don / Blue Ka), interactive scrubber timeline with time display, and audio element synchronization.
- **Waveform Unit Testing:**
  - Added `test_audio_waveform_endpoint` in `tests/test_pipeline.py` verifying envelope structure, note timestamps, and duration calculation.
- **Automated Frontend Verification:**
  - Updated Playwright test `tests/verify_frontend_interactions.py` validating waveform canvas visibility, metadata loading (`597 notes`, `93.15s`), and scrubber click seeking.

## [1.6.0] - 2025-11-08
### Added
- **Live WebSocket Microphone Audio Streaming:**
  - Added `@app.websocket("/ws/stream")` endpoint in `dashboard.py` accepting binary Float32 PCM audio streaming chunks directly from client microphones.
  - Implemented continuous streaming onset detection and classification via `StreamingOnsetDetector` with instantaneous JSON onset trigger event notifications (`DON`, `KA`, confidence, energy, spectral centroid).
  - Designed interactive UI controls in the Unified Studio with WebAudio API microphone input streaming, dynamic connection indicators, live onset badges with color animations (Red Don / Blue Ka), and detailed status feedback.
- **WebSocket Streaming Unit Test:**
  - Added `test_websocket_audio_stream` in `tests/test_pipeline.py` verifying binary PCM communication, onset event dispatch, and connection lifecycle.
- **Frontend Interaction Verification:**
  - Created automated Playwright verification script `tests/verify_frontend_interactions.py` ensuring responsive interaction, error-free DOM state, and visual UI fidelity.

## [1.5.0] - 2025-11-08
### Added
- **Unified CLI Entry Point (`run.py`):**
  - Added clean, standard CLI argument parser with subcommands: `studio`, `train`, `infer`, `synthesize`, `export-onnx`, and `stream-sim`.
  - Enables direct scripted or interactive execution of any stage in the neural onset pipeline.
- **CLI Test Suite:**
  - Added `tests/test_cli.py` covering main and subcommand help parameters.

## [1.4.0] - 2025-11-08
### Added
- **Real-Time Sliding-Window Audio Chunk Streaming (`audio_stream.py`):**
  - Added `AudioChunkStreamer` simulating live audio buffer feeds (1024 samples/chunk) for arbitrary duration.
  - Implemented `StreamingOnsetDetector` utilizing circular ring buffers, short-term RMS energy ratio calculation, and dynamic spectral centroid analysis to detect and classify Don (low-frequency kick) vs. Ka (high-frequency rim/cymbal) onsets on the fly.
- **Custom User Audio File Upload:**
  - Added backend endpoint `POST /api/upload_audio` allowing users to upload arbitrary `.wav` or `.ogg` tracks into the studio pipeline (`data/custom_audio.wav`).
  - Added backend endpoint `POST /api/action/simulate_stream` to run low-latency streaming simulations across either built-in test tracks or custom uploaded tracks.
- **Unified Single-Page Dashboard Integration:**
  - Added interactive card: "🎙️ Live Audio Input & Real-Time Stream Simulator" featuring instant file uploading, source selection, and stream execution.
  - Added real-time visual status badges, interactive tooltips, and diagnostic log notifications.
- **Comprehensive Unit Testing:**
  - Added tests `test_audio_streamer_and_detector` and `test_custom_audio_upload_endpoint` to `tests/test_pipeline.py`.

## [1.3.0] - 2025-11-08
### Added
- **ONNX Model Exporter & Inference Runtime (`export_onnx.py`):**
  - Added `export_to_onnx()` converting PyTorch CNN weights to standardized ONNX graphs (`don_model.onnx`, `ka_model.onnx`).
  - Implemented `ONNXOnsetDetector` using ONNX Runtime for low-latency, CPU-optimized batch inference.
- **Dashboard Dual-Engine Support:**
  - Added Execution Engine selector (PyTorch vs. ONNX Runtime) in the unified web studio.
  - Added direct `/api/action/export_onnx` endpoint and "Export ONNX" action button.
- **Unit Testing:**
  - Added unit test `test_onnx_export_and_runtime` verifying ONNX graph export, shape matching, and probability output ranges.

## [1.2.0] - 2025-11-08
### Added
- **Beat Saber Exporter (.dat v2 schema):** Automated generation of Beat Saber difficulty chart files mapping Don (Red saber, bottom layer down cut) and Ka (Blue saber, middle layer up cut) with quantized beat timestamps.
- **Beat Saber Dashboard Download:** Added `/api/download/beatsaber` endpoint and web UI download button with descriptive tooltips.
- **Extended Test Suite:** Added unit test coverage for Beat Saber `.dat` JSON schema and note layout.

## [1.1.0] - 2025-11-08
### Added
- **Multi-Format Rhythm Game Chart Exporters (`chart_exporters.py`):**
  - **osu! Taiko (.osu Mode 1):** Full timing point and hitObject generation mapping Don to normal hitsounds and Ka to whistle/clap hitsounds.
  - **StepMania (.sm):** Quantized 16th-note measure generation for dance-single charts mapping Don to Left+Right and Ka to Up+Down.
- **Dashboard Multi-Format Integration:** Added download buttons and direct endpoints (`/api/download/osu`, `/api/download/sm`) in the Studio UI.
- **Unit Tests:** Extended pipeline tests in `tests/test_pipeline.py` with multi-format chart verification.

## [1.0.0] - 2025-11-08
### Added
- **Unified Single-Page Dashboard (`dashboard.py`):** Complete end-to-end studio for Mel-spectrogram preprocessing, CNN training (Don/Ka), neural onset inference, peak picking, and audio/chart synthesis.
- **System Tray Controller (`tray_app.py`):** Desktop tray controller for background server lifecycle control, browser launching, and clean shutdown.
- **Automated Test Suite (`tests/test_pipeline.py`):** Model forward pass, signal smoothing, peak picking, and dashboard API testing.
- **Governance & Standards Documentation:** Initialized `VERSION.md`, `ROADMAP.md`, `TODO.md`, `VISION.md`, `MEMORY.md`, `DEPLOY.md`, `IDEAS.md`, and `HANDOFF.md`.

### Fixed
- Modernized `librosa.util.peak_pick` in `synthesize.py` with explicit keyword arguments (`pre_max`, `post_max`, `pre_avg`, `post_avg`, `delta`, `wait`).
- Resolved `scipy.signal.blackmanharris` deprecation in `music_processor.py` by transitioning to `scipy.signal.windows.blackmanharris`.
- Fixed Librosa mel filterbank keyword arguments in `music_processor.py`.
- Replaced deprecated `torch.nn.functional.sigmoid` with `torch.sigmoid` in `model.py`.
- Removed unsupported Numba `@jit` decorators on Python `Audio` class objects in `music_processor.py`.
