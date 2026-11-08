# Session Handoff & Continuity Notes

## 1. Session Overview
- Modernized legacy Python/PyTorch/Librosa code to support modern versions without crashes.
- Created standard governance documents (`VERSION.md`, `VISION.md`, `ROADMAP.md`, `TODO.md`, `MEMORY.md`, `DEPLOY.md`, `IDEAS.md`, `CHANGELOG.md`, `HANDOFF.md`).
- Built unified, single-page consolidated Web Dashboard and desktop System Tray application.
- Added comprehensive unit, integration, and Playwright verification tests.
- Extended export capabilities with multi-format chart exporters for osu! (.osu) and StepMania (.sm).

## 2. Key Architecture & File Map
- `dashboard.py`: FastAPI server serving the unified UI, live log streams, and REST API for preprocessing, training, inference, and synthesis.
- `tray_app.py`: System tray manager with server status and graceful shutdown options.
- `model.py`, `music_processor.py`, `synthesize.py`, `infer.py`: Core signal processing, CNN modeling, and chart synthesis pipelines.
- `data/`: Sample test tracks, reduced training dataset, models, and pickled features.

## 3. Next Steps for Successor Models
- Continue expanding browser-based visual chart editing.
- Benchmark inference performance and explore ONNX runtime optimization.
