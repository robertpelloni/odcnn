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

    def test_quantize_timestamps_algorithm(self):
        """Test beat grid timestamp snapping across different musical subdivisions."""
        import numpy as np
        import chart_exporters

        # At 120 BPM, quarter notes occur every 0.5s
        raw = np.array([0.48, 1.02, 1.51, 1.99, 2.53])
        snapped = chart_exporters.quantize_timestamps_to_grid(raw, bpm=120.0, division=4)
        expected = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        self.assertTrue(np.allclose(snapped, expected, atol=1e-3))

        # At 120 BPM with 1/16th subdivision (division=16, step = 0.125s)
        raw_16th = np.array([0.12, 0.26, 0.37, 0.51])
        snapped_16th = chart_exporters.quantize_timestamps_to_grid(raw_16th, bpm=120.0, division=16)
        expected_16th = np.array([0.125, 0.25, 0.375, 0.5])
        self.assertTrue(np.allclose(snapped_16th, expected_16th, atol=1e-3))

    def test_quantize_charts_api_endpoint(self):
        """Test /api/chart/quantize endpoint in dashboard."""
        from fastapi.testclient import TestClient
        import dashboard
        client = TestClient(dashboard.app)

        res = client.post("/api/chart/quantize", data={"bpm": "140.0", "division": "16", "offset": "0.05"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["bpm"], 140.0)
        self.assertEqual(data["division"], 16)
        self.assertIn("don_count", data)
        self.assertIn("ka_count", data)

    def test_export_midi_and_download_endpoint(self):
        """Test Standard MIDI File generator and /api/download/midi endpoint."""
        import numpy as np
        import chart_exporters
        from fastapi.testclient import TestClient
        import dashboard

        class MockSong:
            samplerate = 44100

        # Test direct export_midi generator
        midi_path = Path("/tmp/test_export.mid")
        don = np.array([100, 200, 400])
        ka = np.array([150, 300])
        res_path = chart_exporters.export_midi(
            str(midi_path), MockSong(), don, ka, title="Test MIDI", bpm=128.0
        )
        self.assertTrue(midi_path.exists())
        with open(midi_path, "rb") as f:
            content = f.read()
        self.assertTrue(content.startswith(b"MThd"))
        self.assertIn(b"MTrk", content)
        self.assertIn(b"Test MIDI", content)

        # Test dashboard download endpoint
        client = TestClient(dashboard.app)
        res = client.get("/api/download/midi")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "audio/midi")
        self.assertTrue(res.content.startswith(b"MThd"))

    def test_stem_separation_and_api(self):
        """Test Harmonic/Percussive source separation and stems endpoints."""
        import numpy as np
        import stem_separator
        from fastapi.testclient import TestClient
        import dashboard

        # 1. Test separate_hpss math on synthetic tone + clicks
        sr = 44100
        t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
        sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        click = np.zeros_like(t)
        click[5000:5050] = 0.9

        mix = sine + click
        harm, perc, h_ratio, p_ratio = stem_separator.separate_hpss(mix, sr=sr, kernel_size=15)
        self.assertEqual(len(harm), len(mix))
        self.assertEqual(len(perc), len(mix))
        self.assertGreater(h_ratio, 0.0)
        self.assertGreater(p_ratio, 0.0)

        # 2. Test subband separation
        bass, mid, treble = stem_separator.separate_subbands(mix, sr=sr)
        self.assertEqual(len(bass), len(mix))
        self.assertEqual(len(mid), len(mix))
        self.assertEqual(len(treble), len(mix))

        # 3. Test stems API endpoints
        client = TestClient(dashboard.app)
        res_status = client.get("/api/stems/status")
        self.assertEqual(res_status.status_code, 200)
        self.assertIn("available", res_status.json())

        # Test download stem endpoints (percussive exists from setup)
        res_perc = client.get("/api/download/stem/percussive")
        if res_perc.status_code == 200:
            self.assertEqual(res_perc.headers["content-type"], "audio/wav")
            self.assertGreater(len(res_perc.content), 1000)

        # Test invalid stem type returns 400
        res_inv = client.get("/api/download/stem/invalid_stem")
        self.assertEqual(res_inv.status_code, 400)

if __name__ == "__main__":
    unittest.main()
