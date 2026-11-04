# IDEAS: Explorations, Refactoring & Feature Expansion

1. **CRNN / Attention Multi-Note Architecture:**
   - Instead of training separate models for `don` and `ka`, train a multi-task network with joint feature extraction and multi-head outputs for onset probability and note classification.
   - Introduce bidirectional GRU/LSTM layers after CNN pooling to capture rhythmic meter context and syncopation.

2. **Full In-Browser Interactive Chart Editor:**
   - Integrate a visual Taiko highway editor (HTML5 Canvas / WebGL) directly inside the unified dashboard.
   - Allow users to click and drag note markers, adjust thresholds in real time with instantaneous visual feedback, and test-play songs right inside the browser.

3. **ONNX Export and WebAssembly Inference:**
   - Export PyTorch `convNet` weights to ONNX format.
   - Run inference client-side using `onnxruntime-web` or WebAssembly, enabling zero-install chart generation directly in the user's browser without needing a GPU server backend.

4. **Automated Beatmap Difficulty Generator:**
   - Generate all four Taiko difficulty levels (Easy, Normal, Hard, Oni) from a single inferred song by dynamically thinning notes based on onset confidence and local musical density.
