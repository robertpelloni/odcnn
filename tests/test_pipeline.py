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

if __name__ == "__main__":
    unittest.main()
