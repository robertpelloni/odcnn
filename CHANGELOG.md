# CHANGELOG

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
