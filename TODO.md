# Granular Short-Term Tasks & Immediate Fixes

- [x] Fix `TypeError: peak_pick() takes 1 positional argument but 7 were given` in `synthesize.py`.
- [x] Fix `numba.core.errors.TypingError` on custom Audio objects in `music_processor.py`.
- [x] Fix `AttributeError: module 'scipy.signal' has no attribute 'blackmanharris'` in `music_processor.py`.
- [x] Replace deprecated `F.sigmoid` with `torch.sigmoid` in `model.py`.
- [x] Modernize `requirement.txt` with working pinned dependencies.
- [x] Implement unified Dashboard application `dashboard.py` with FastAPI and responsive frontend.
- [x] Implement system tray controller `tray_app.py`.
- [x] Add automated test suite `tests/test_pipeline.py`.
- [x] Complete Playwright frontend testing and UI screenshot verification.
- [x] Implement multi-format rhythm game chart exporters for osu! and StepMania (`chart_exporters.py`).
- [x] Add dashboard download endpoints and buttons for `.osu` and `.sm` files.
- [x] Implement Beat Saber exporter (`export_beat_saber`) in `chart_exporters.py` and UI download.
- [x] Implement ONNX export and ONNX Runtime inference engine (`export_onnx.py`).
- [x] Integrate dual inference engines (PyTorch & ONNX Runtime) in `dashboard.py`.
- [ ] Add real-time microphone stream input for live onset inference.
