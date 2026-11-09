# CHANGELOG

## [2.4.0] - 2025-11-08
### Added
- **Sustained Drumroll & Balloon Detection Engine (`roll_detector.py`):**
  - Designed and implemented continuous RMS persistence and spectral flux variance suppression algorithm to accurately detect sustained rolls and held notes in musical audio.
  - Automatically classifies sustained sounds based on mean spectral centroid and RMS energy into:
    - **Yellow Drumroll (TJA `5`):** Standard sustained drumroll for standard energy levels.
    - **Big Drumroll (TJA `6`):** High-amplitude sustained roll for prominent drumrolls.
    - **Balloon Note (TJA `7`):** High-flux bursting rolls with designated pop thresholds.
    - **Roll End Marker (TJA `8`):** Exact temporal termination marker for chart parsers.
  - Implemented onset deduplication helper `merge_onsets_and_rolls()` which deduplicates discrete Don/Ka note onsets occurring inside sustained roll intervals to prevent note overlapping in rhythm charts.
- **Synthesizer & Multi-Format Exporter Integration:**
  - Integrated `detect_sustained_rolls()` and `merge_onsets_and_rolls()` into `synthesize.py` (`create_tja`), generating accurate `5`, `6`, `7`, and `8` markers in `.tja` measure blocks.
  - Updated `chart_exporters.py` to map detected sustained rolls to Mode 1 spinner hitObjects in osu! Taiko (`.osu`) with millisecond start/end timestamps.
- **FastAPI Studio REST API Integration:**
  - Added `POST /api/rolls/detect` endpoint accepting min duration (seconds) and energy ratio thresholds, returning detected rolls with note codes and formatted millisecond timestamps.
  - Updated `GET /api/audio_waveform` to parse and return detected sustained roll intervals (`rolls`) alongside regular notes and waveform envelopes.
- **Unified Studio Dashboard UI Controls:**
  - Added interactive card: "🥁 Sustained Drumroll & Balloon Detector" (`#rollsCard`) to the single-page studio layout.
  - Configurable minimum roll duration (`#rollMinDuration`) and energy ratio (`#rollEnergyRatio`) controls.
  - "Detect Sustained Rolls" action button (`#btnDetectRolls`) and dynamic real-time status badge (`#rollsStatusBadge`).
  - Interactive table displaying detected rolls with note type badges (Yellow, Big, Balloon) and start/end time offsets.
- **Automated Test Suite & Frontend Playwright Verification:**
  - Added unit test `test_roll_detection_and_api` in `tests/test_pipeline.py` verifying synthetic sustained tone detection, onset deduplication, and FastAPI REST endpoint responses (expanding test suite to 26 passing tests).
  - Updated automated Playwright test `tests/verify_frontend_interactions.py` with sustained roll detection button interaction and status verification, confirmed visually via snapshot inspection.

## [2.3.0] - 2025-11-08
### Added
- **Dynamic Multi-Stem Audio Separation Engine (`stem_separator.py`):**
  - Designed and implemented pure-Python, zero-dependency Harmonic-Percussive Source Separation (HPSS) using 2D median filtering across STFT spectral representations (`separate_hpss`).
  - Separates transient rhythmic drum percussions (vertical spectral lines) from tonal melodic instruments and vocals (horizontal spectral lines) using soft spectral masking.
  - Implemented 3-band sub-band spectral separation (`separate_subbands`) utilizing 4th-order Butterworth filters to isolate Bass (< 250 Hz), Mid (250 Hz - 2500 Hz), and Treble (> 2500 Hz) stems.
  - Implemented automated batch stem extraction and 16-bit WAV writing (`extract_and_save_stems`) outputting `stem_percussive.wav`, `stem_harmonic.wav`, `stem_bass.wav`, and `stem_treble.wav` to `data/stems/`.
- **FastAPI Studio REST API Integration:**
  - Added `POST /api/stems/extract` triggering asynchronous or synchronous multi-stem extraction for either test audio or custom uploaded audio.
  - Added `GET /api/stems/status` reporting stem generation status, timestamps, and available stem files.
  - Added `GET /api/download/stem/{stem_type}` allowing instant downloading and streaming of percussive, harmonic, bass, or treble stems with `audio/wav` headers.
- **Unified Studio Dashboard UI Controls:**
  - Added interactive card: "🎛️ Dynamic Audio Stems & Harmonic/Percussive Separator" (`#stemsCard`) within the consolidated studio interface.
  - Included audio track source selector, "Extract Audio Stems" trigger button (`#btnExtractStems`), real-time extraction status badge (`#stemsStatusBadge`), and independent HTML5 audio preview players and download buttons for percussive and harmonic stems.
- **Automated Test Suite & Frontend Verification:**
  - Added comprehensive unit test `test_stem_separation_and_api` in `tests/test_pipeline.py` verifying mathematical separation energy preservation, file persistence, REST API responses, and download stream endpoints (test suite expanded to 25/25 passing tests).
  - Updated Playwright interaction test `tests/verify_frontend_interactions.py` with stem separation testing, verified cleanly without errors, and confirmed with visual inspection of UI snapshot.

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
