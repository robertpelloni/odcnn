# HANDOFF: Session Memory & Operational Status

## Current System State (v1.5.0)
- **ODCNN Engine Status:** Fully functional, modernized for Python 3.12, PyTorch 2.x, Librosa 0.10.x, SciPy 1.14+, and ONNX Runtime.
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
- **Automated Tests:** 13/13 unit tests passing via `python3 -m unittest discover tests`.
- **Frontend Verification:** Screenshot verified with Playwright at `/home/jules/verification/verification.png`.

## File Reference & Structure
- `run.py`: Unified CLI entry point for all operations.
- `synthesize.py`: Signal smoothing, multi-parameter peak picking, Taiko audio synthesis, and multi-format chart generation.
- `music_processor.py`: STFT and Mel-frequency filterbank feature extraction.
- `model.py`: Schlüter & Böck CNN architecture for onset probabilities.
- `chart_exporters.py`: Chart writers for osu!, StepMania, Beat Saber.
- `export_onnx.py`: ONNX exporter and ONNX Runtime inference class.
- `audio_stream.py`: Sliding buffer audio stream generator and real-time ring buffer onset detector.
- `dashboard.py`: Unified FastAPI single-page dashboard with real-time logging, metrics, actions, dual engine support, and audio upload/streaming simulator card.
- `tray_app.py`: System tray manager with desktop process management and shutdown.
- `tests/test_pipeline.py`: Comprehensive test suite (11 unit tests).
- `tests/test_cli.py`: CLI entry point test suite (2 unit tests).
- `VERSION.md`: Sourced single version string (`1.5.0`).

## Next Suggested Architectural Work
- WebRTC / WebSocket low-latency audio streaming directly from browser microphone input.
- Conformer / Recurrent network architecture exploration for long-range tempo awareness.
- Standalone C++ runner utilizing exported ONNX models for embedded arcade rhythm hardware.
