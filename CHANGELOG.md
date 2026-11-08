# CHANGELOG

All notable changes to this project are documented here.
Version format follows Semantic Versioning sourced from `VERSION.md`.

## [1.1.0] - 2026-06-24
### Added
- Multi-format chart exporters in `chart_exporters.py` for osu! Taiko (`.osu` Mode 1) and StepMania (`.sm` dance-single 4-panel).
- API download endpoints `/api/download/osu` and `/api/download/sm` in `dashboard.py`.
- Download buttons and interactive format management directly inside the unified dashboard.
- Unit tests verifying multi-format chart exports in `tests/test_pipeline.py`.

## [1.0.0] - 2026-06-24
### Added
- Standardized autonomous governance documentation: `VERSION.md`, `VISION.md`, `ROADMAP.md`, `TODO.md`, `MEMORY.md`, `DEPLOY.md`, `IDEAS.md`, `CHANGELOG.md`, `HANDOFF.md`.
- Unified single-page web dashboard (`dashboard.py`) built with FastAPI, real-time logging, interactive controls, audio synthesis, and chart export.
- Desktop System Tray controller (`tray_app.py`) for background server management.
- Comprehensive end-to-end test suite (`tests/test_pipeline.py`) validating all signal processing, model inference, and API endpoints.

### Changed
- Modernized `music_processor.py` to use keyword arguments for `librosa.filters.mel`.
- Updated `music_processor.py` to support `scipy.signal.windows.blackmanharris`.
- Removed unsupported Numba nopython object decoration in `multi_fft_and_melscale`.
- Updated `model.py` to use `torch.sigmoid` instead of deprecated `F.sigmoid`.
- Modernized `synthesize.py` to use keyword arguments for `librosa.util.peak_pick`.
- Updated `requirement.txt` with all necessary packages.
