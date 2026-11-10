import unittest
import torch
import numpy as np
from pathlib import Path
from model_crnn import CRNNOnsetMaxNet, RhythmAttention
import export_onnx

class TestCRNNPipeline(unittest.TestCase):
    def test_rhythm_attention_shape(self):
        batch, seq_len, hidden_dim = 2, 15, 64
        attn = RhythmAttention(hidden_dim=hidden_dim, num_heads=4)
        x = torch.randn(batch, seq_len, hidden_dim)
        out = attn(x)
        self.assertEqual(out.shape, (batch, seq_len, hidden_dim))

    def test_crnn_forward_pass_and_bounds(self):
        model = CRNNOnsetMaxNet()
        dummy_input = torch.randn(4, 3, 80, 15)
        out = model(dummy_input)
        self.assertEqual(out.shape, (4, 1))
        # Ensure sigmoid output bounds [0.0, 1.0]
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_crnn_gradient_flow(self):
        model = CRNNOnsetMaxNet()
        dummy_input = torch.randn(2, 3, 80, 15, requires_grad=True)
        target = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        criterion = torch.nn.MSELoss()

        output = model(dummy_input, istraining=True)
        loss = criterion(output, target)
        loss.backward()

        self.assertIsNotNone(dummy_input.grad)
        self.assertIsNotNone(model.conv1.weight.grad)
        self.assertIsNotNone(model.gru.weight_ih_l0.grad)
        self.assertIsNotNone(model.fc2.weight.grad)

    def test_crnn_feature_infer(self):
        model = CRNNOnsetMaxNet()
        device = torch.device("cpu")
        # Fake feature spectrogram: 3 channels, 80 mel bands, 45 time frames
        fake_feats = np.random.randn(3, 80, 45).astype(np.float32)
        out = model.infer(fake_feats, device=device, minibatch=8)
        # Expected frames: total_frames - soundlen = 45 - 15 = 30
        self.assertEqual(len(out), 30)
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))

    def test_crnn_onnx_export(self):
        pth_path = Path("models/don_crnn.pth")
        onnx_path = Path("models/don_crnn_test.onnx")
        export_onnx.export_to_onnx(pth_path, onnx_path, arch="crnn")
        self.assertTrue(onnx_path.exists())
        self.assertGreater(onnx_path.stat().st_size, 1000)

        # Verify ONNX Runtime inference
        detector = export_onnx.ONNXOnsetDetector(onnx_path)
        fake_feats = np.random.randn(3, 80, 30).astype(np.float32)
        res = detector.infer(fake_feats, minibatch=8)
        self.assertEqual(len(res), 15)
        # Clean up temporary test file
        if onnx_path.exists():
            onnx_path.unlink()

if __name__ == "__main__":
    unittest.main()
