# HANDOFF: Session Memory & Operational Status

## Current System State (v1.8.0)
- **ODCNN Engine Status:** Fully functional, modernized for Python 3.12, PyTorch 2.x, Librosa 0.10.x, SciPy 1.14+, and ONNX Runtime.
- **Mel-Frequency Spectrogram Heatmap Visualizer:**
  - Spectrogram data endpoint `/api/spectrogram_data` computing 48-mel frequency bins across downsampled STFT frames normalized [0.0, 1.0].
  - Interactive HTML5 Canvas rendering of spectral energy using Magma/Inferno color ramp (purple -> magenta -> orange -> yellow -> white) with frequency axis markings.
  - Synchronized two-way scrubbing linking spectrogram, waveform, and audio playback.
- **Interactive Audio Waveform Visualizer & Rhythm Onset Scrubber:**
  - Fast downsampled dual min/max envelope generator endpoint `/api/audio_waveform` returning 1000 point waveform profile and parsed rhythm note timestamps.
  - Interactive HTML5 Canvas visualizer rendering waveforms with overlaid color-coded hit markers (Red Don / Blue Ka) and synchronized timeline scrubber cursor.
  - Click-to-scrub seeks the studio `<audio>` element with real-time timecode updates.
- **Live WebSocket Microphone Streaming (`/ws/stream`):**
  - FastAPI WebSocket endpoint accepts Float32 binary PCM audio chunks directly from user microphones.
  - Streaming onset detector evaluates RMS energy jumps and classifies hits as Don or Ka based on spectral centroid frequencies.
  - Studio UI includes live microphone start/stop toggle, connection indicators, and animated visual hit badges (Red Don / Blue Ka).
- **Unified CLI Runner (`run.py`):** Consolidated command line interface supporting `studio`, `train`, `infer`, `synthesize`, `export-onnx`, and `stream-sim`.
- **Unified Web Studio:** Running on `http://0.0.0.0:8000` via FastAPI (`dashboard.py`).
- **Desktop System Tray Controller:** Implemented in `tray_app.py` for headless background server management.
- **Rhythm Game Exporters (`chart_exporters.py`):**
  - Taiko no Tatsujin (`.tja`)
  - osu! Taiko Mode 1 (`.osu`)
  - StepMania Dance Single (`.sm`)
  - Beat Saber (`.dat` v2 schema)
- **ONNX Acceleration Engine (`export_onnx.py`):**
  - Graph exporter for Don/Ka models (`models/don_model.onnx`, `models/ka_model.onnx`).
  - ONNX Runtime execution engine integrated with UI engine selection toggle.
- **Live Audio Streaming & Upload Pipeline (`audio_stream.py`):**
  - `AudioChunkStreamer` and `StreamingOnsetDetector` circular buffer onset detector.
  - Custom audio track upload endpoint (`POST /api/upload_audio`).
  - Real-time sliding buffer onset simulation endpoint (`POST /api/action/simulate_stream`).
- **Automated Tests:** 16/16 unit tests passing via `python3 -m unittest discover tests`.
- **Frontend Verification:** Playwright frontend interaction test verified and confirmed via visual inspection of `/home/jules/verification/verification.png`.

## File Reference & Structure
- `run.py`: Unified CLI entry point for all operations.
- `synthesize.py`: Signal smoothing, multi-parameter peak picking, Taiko audio synthesis, and multi-format chart generation.
- `music_processor.py`: STFT and Mel-frequency filterbank feature extraction.
- `model.py`: Schlüter & Böck CNN architecture for onset probabilities.
- `chart_exporters.py`: Chart writers for osu!, StepMania, Beat Saber.
- `export_onnx.py`: ONNX exporter and ONNX Runtime inference class.
- `audio_stream.py`: Sliding buffer audio stream generator and real-time ring buffer onset detector.
- `dashboard.py`: Unified FastAPI single-page dashboard with real-time logging, metrics, actions, dual engine support, audio upload/streaming simulator, WebSocket live mic streaming, interactive waveform scrubber, and Mel spectrogram heatmap.
- `tray_app.py`: System tray manager with desktop process management and shutdown.
- `tests/test_pipeline.py`: Comprehensive test suite (14 unit tests including waveform, spectrogram, and WebSocket tests).
- `tests/test_cli.py`: CLI entry point test suite (2 unit tests).
- `tests/verify_frontend_interactions.py`: Playwright frontend interaction test script.
- `VERSION.md`: Sourced single version string (`1.8.0`).

## Next Suggested Architectural Work
- BPM tap tempo & beat grid snapping tool for chart editing.
- Conformer / Recurrent network architecture exploration for long-range tempo awareness.
- Standalone C++ runner utilizing exported ONNX models for embedded arcade rhythm hardware.
