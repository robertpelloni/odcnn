import unittest
import torch
import numpy as np
import os
from pathlib import Path
from fastapi.testclient import TestClient

import model
import music_processor
import synthesize
from dashboard import app

class TestODCNNPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_model_forward_pass(self):
        """Test CNN forward pass dimensions with dummy spectrogram tensor."""
        net = model.convNet()
        # Input shape: (batch_size=2, channels=3, freq_bins=80, time_frames=15)
        dummy_input = torch.randn(2, 3, 80, 15)
        out = net(dummy_input, istraining=False, minibatch=2)
        self.assertEqual(out.shape, (2, 1))
        # Ensure sigmoid output is in range [0, 1]
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_smooth_function(self):
        """Test smoothing window on synthetic 1D signal."""
        x = np.sin(np.linspace(0, 10, 100))
        smoothed = synthesize.smooth(x, window_len=5)
        # scipy mode='valid' with mirrored padding creates length = len(x) + window_len - 1
        self.assertGreater(len(smoothed), 0)

    def test_peak_picking_synthesize(self):
        """Test modernized peak picking with synthetic peaks."""
        signal_data = np.zeros(100)
        signal_data[20] = 1.0
        signal_data[50] = 0.9
        peaks = synthesize.peak_pick(
            signal_data,
            pre_max=1,
            post_max=1,
            pre_avg=2,
            post_avg=2,
            delta=0.1,
            wait=3
        )
        self.assertIn(20, peaks)
        self.assertIn(50, peaks)

    def test_dashboard_api_status(self):
        """Test /api/status endpoint response."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("version", data)
        self.assertIn("cpu_usage", data)
        self.assertIn("ram_usage", data)
        self.assertIn("artifacts", data)

    def test_dashboard_api_logs(self):
        """Test /api/logs endpoint response."""
        response = self.client.get("/api/logs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("logs", data)

    def test_dashboard_index_html(self):
        """Test HTML rendering for the single-page unified UI."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ODCNN AUTOPILOT STUDIO", response.text)
        self.assertIn("Step 3: Peak Picking & Taiko Chart Synthesis", response.text)
        self.assertIn(".OSU", response.text)
        self.assertIn(".SM", response.text)

    def test_multi_format_chart_exporters(self):
        """Test export_osu, export_sm, and export_beat_saber file structure."""
        from chart_exporters import export_osu, export_sm, export_beat_saber
        import json
        class DummySong:
            samplerate = 44100
        song = DummySong()
        don_ts = np.array([100, 200])
        ka_ts = np.array([150, 250])

        osu_file = "/tmp/test_chart.osu"
        export_osu(osu_file, song, don_ts, ka_ts, title="Test Song", bpm=120)
        with open(osu_file, "r") as f:
            osu_content = f.read()
        self.assertIn("Mode: 1", osu_content)
        self.assertIn("[HitObjects]", osu_content)

        sm_file = "/tmp/test_chart.sm"
        export_sm(sm_file, song, don_ts, ka_ts, title="Test Song", bpm=120)
        with open(sm_file, "r") as f:
            sm_content = f.read()
        self.assertIn("#TITLE:Test Song;", sm_content)
        self.assertIn("#NOTES:", sm_content)

        dat_file = "/tmp/test_chart.dat"
        export_beat_saber(dat_file, song, don_ts, ka_ts, bpm=120)
        with open(dat_file, "r") as f:
            bs_data = json.load(f)
        self.assertEqual(bs_data["_version"], "2.2.0")
        self.assertIn("_notes", bs_data)
        self.assertEqual(len(bs_data["_notes"]), 4)
        # Check note keys
        first_note = bs_data["_notes"][0]
        self.assertIn("_time", first_note)
        self.assertIn("_lineIndex", first_note)
        self.assertIn("_lineLayer", first_note)
        self.assertIn("_type", first_note)
        self.assertIn("_cutDirection", first_note)

    def test_onnx_export_and_runtime(self):
        """Test ONNX export and ONNX Runtime inference detector."""
        import export_onnx
        pth_path = Path("models/don_model.pth")
        if pth_path.exists():
            test_onnx = Path("/tmp/test_don.onnx")
            export_onnx.export_to_onnx(pth_path, test_onnx)
            self.assertTrue(test_onnx.exists())

            detector = export_onnx.ONNXOnsetDetector(test_onnx)
            # Create synthetic feats: [3, 80, 30]
            dummy_feats = np.random.randn(3, 80, 30).astype(np.float32)
            res = detector.infer(dummy_feats, soundlen=15, minibatch=8)
            # Total frames = 30 - 15 = 15
            self.assertEqual(len(res), 15)
            self.assertTrue(np.all(res >= 0.0) and np.all(res <= 1.0))



    def test_audio_streamer_and_detector(self):
        """Test real-time streaming chunk generator and onset detector."""
        import audio_stream
        # Generate 1 second of synthetic audio containing a loud click at 0.5s
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = 0.01 * np.sin(2 * np.pi * 440 * t)
        # Add a drum hit spike around index 22050
        signal[22000:22200] += 0.8 * np.sin(2 * np.pi * 150 * np.linspace(0, 0.005, 200))

        streamer = audio_stream.AudioChunkStreamer(signal, sample_rate=sr, chunk_size=512)
        detector = audio_stream.StreamingOnsetDetector(sample_rate=sr, energy_threshold=0.03)

        detected = []
        for c_idx, t_sec, chunk in streamer.stream_chunks():
            onsets = detector.push_chunk(chunk)
            if onsets:
                detected.extend(onsets)

        self.assertGreater(len(detected), 0)
        self.assertIn("timestamp", detected[0])
        self.assertIn("type", detected[0])
        self.assertIn(detected[0]["type"], ["don", "ka"])

    def test_custom_audio_upload_endpoint(self):
        """Test /api/upload_audio endpoint."""
        import io
        fake_wav = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
        response = self.client.post(
            "/api/upload_audio",
            files={"file": ("custom.wav", fake_wav, "audio/wav")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], "custom.wav")

    def test_simulate_stream_endpoint(self):
        """Test /api/action/simulate_stream endpoint starts background worker without errors."""
        response = self.client.post(
            "/api/action/simulate_stream",
            data={"energy_threshold": "0.05", "chunk_size": "2048", "audio_source": "sample"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)

    def test_websocket_audio_stream(self):
        """Test WebSocket /ws/stream binary PCM chunk intake and onset emission."""
        with self.client.websocket_connect("/ws/stream") as websocket:
            # Send silent chunk
            silent_chunk = np.zeros(1024, dtype=np.float32)
            websocket.send_bytes(silent_chunk.tobytes())

            # Send spike chunk simulating a drum hit
            t = np.linspace(0, 0.02, 1024, endpoint=False)
            hit_chunk = (0.8 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
            websocket.send_bytes(hit_chunk.tobytes())

            # Receive onset response
            data = websocket.receive_json()
            self.assertIn("onsets", data)
            self.assertGreater(len(data["onsets"]), 0)
            self.assertIn("type", data["onsets"][0])
            self.assertIn(data["onsets"][0]["type"], ["don", "ka"])

    def test_audio_waveform_endpoint(self):
        """Test /api/audio_waveform returns valid downsampled envelope points and note timestamps."""
        response = self.client.get("/api/audio_waveform")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("source", data)
        self.assertIn("duration", data)
        self.assertGreater(data["duration"], 0)
        self.assertIn("points", data)
        self.assertEqual(len(data["pos"]), data["points"])
        self.assertEqual(len(data["neg"]), data["points"])
        self.assertIn("notes", data)
        self.assertIsInstance(data["notes"], list)

    def test_spectrogram_data_endpoint(self):
        """Test /api/spectrogram_data returns normalized 2D mel-frequency spectral matrix."""
        response = self.client.get("/api/spectrogram_data?n_mels=32")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("source", data)
        self.assertIn("duration", data)
        self.assertGreater(data["duration"], 0)
        self.assertEqual(data["n_mels"], 32)
        self.assertIn("matrix", data)
        self.assertEqual(len(data["matrix"]), 32)
        self.assertGreater(len(data["matrix"][0]), 0)
        # Verify values are in normalized range [0.0, 1.0]
        val = data["matrix"][0][0]
        self.assertTrue(0.0 <= val <= 1.0)

if __name__ == "__main__":
    unittest.main()
