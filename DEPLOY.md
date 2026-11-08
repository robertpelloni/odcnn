# DEPLOYMENT AND ENVIRONMENT SETUP

## Requirements
- Python 3.10+ (tested on Python 3.12)
- Audio codec support: libsndfile
- PyTorch with CPU or CUDA support

## Quickstart Setup
```bash
# Clone the repository
git clone https://github.com/robertpelloni/odcnn.git
cd odcnn

# Install required dependencies
pip install -r requirement.txt
```

## Running via Unified CLI Runner (`run.py`)
Launch the unified studio:
```bash
python3 run.py studio --port 8000
```
Then navigate your browser to `http://localhost:8000`.

Or execute pipeline commands directly:
```bash
# Train Don/Ka models
python3 run.py train --target don
python3 run.py train --target ka

# Run neural inference
python3 run.py infer --target both

# Synthesize charts (TJA, OSU, SM, Beat Saber)
python3 run.py synthesize --delta 0.05 --wait 3

# Export PyTorch weights to portable ONNX graphs
python3 run.py export-onnx

# Simulate real-time streaming sliding buffer onset detection
python3 run.py stream-sim --chunk-size 1024
```

## Running the System Tray Icon
On desktop environments with a GUI display:
```bash
python3 tray_app.py
```
This enables launching the dashboard in the default browser, monitoring health, and gracefully shutting down the background server.
