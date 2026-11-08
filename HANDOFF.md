# HANDOFF: Session Memory & Operational Status

## Current System State (v1.3.0)
- **ODCNN Engine Status:** Fully functional, modernized for Python 3.12, PyTorch 2.x, Librosa 0.10.x, SciPy 1.14+, and ONNX Runtime.
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
- **Automated Tests:** 8/8 unit tests passing via `python3 -m unittest discover tests`.
- **Frontend Verification:** Screenshot verified with Playwright at `tests/dashboard_screenshot.png`.

## File Reference & Structure
- `synthesize.py`: Signal smoothing, multi-parameter peak picking, Taiko audio synthesis, and multi-format chart generation.
- `music_processor.py`: STFT and Mel-frequency filterbank feature extraction.
- `model.py`: Schlüter & Böck CNN architecture for onset probabilities.
- `chart_exporters.py`: Chart writers for osu!, StepMania, Beat Saber.
- `export_onnx.py`: ONNX exporter and ONNX Runtime inference class.
- `dashboard.py`: Unified FastAPI single-page dashboard with real-time logging, metrics, actions, and dual engine support.
- `tray_app.py`: System tray manager with desktop process management and shutdown.
- `tests/test_pipeline.py`: Comprehensive test suite (8 tests).
- `VERSION.md`: Sourced single version string (`1.3.0`).

## Next Suggested Architectural Work
- Integrate live streaming microphone input (PortAudio/WebAudio) for zero-latency real-time onset detection.
- Add standalone C++ runner utilizing exported ONNX models for embedded hardware.
