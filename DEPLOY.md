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

## Running the Unified Web Dashboard
Launch the unified dashboard:
```bash
python3 dashboard.py --port 8000
```
Then navigate your browser to `http://localhost:8000`.

## Running the System Tray Icon
On desktop environments with a GUI display:
```bash
python3 tray_app.py
```
This enables launching the dashboard in the default browser, monitoring health, and gracefully shutting down the background server.

## CLI Usage (Headless / Scripting)
1. **Prepare Audio Dataset:**
   ```bash
   python3 music_processor.py reduced  # or 'train', 'test', 'val'
   ```
2. **Train Models:**
   ```bash
   python3 train.py don
   python3 train.py ka
   ```
3. **Run Inference:**
   ```bash
   python3 infer.py don
   python3 infer.py ka
   ```
4. **Synthesize Chart & Audio:**
   ```bash
   python3 synthesize.py
   ```
