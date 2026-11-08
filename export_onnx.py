"""
ONNX Model Exporter & Portable Inference Runtime for ODCNN.
Allows exporting trained PyTorch CNN weights to standardized ONNX graphs
and executing high-throughput, low-latency inference using ONNX Runtime.
"""

import os
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model

class ExportableCNN(nn.Module):
    """
    ONNX-friendly wrapper around ODCNN convNet.
    Matches layer architecture of model.convNet:
    conv1: Conv2d(3, 10, (3, 7))
    conv2: Conv2d(10, 20, 3)
    fc1: Linear(1120, 256)
    fc2: Linear(256, 120)
    fc3: Linear(120, 1)
    """
    def __init__(self, base_model: model.convNet):
        super(ExportableCNN, self).__init__()
        self.conv1 = base_model.conv1
        self.conv2 = base_model.conv2
        self.fc1 = base_model.fc1
        self.fc2 = base_model.fc2
        self.fc3 = base_model.fc3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1: Conv -> ReLU -> MaxPool(3, 1)
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1))

        # Layer 2: Conv -> ReLU -> MaxPool(3, 1)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1))

        # Reshape / Flatten dynamically: shape [B, 1120]
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return torch.sigmoid(self.fc3(x))


def export_to_onnx(
    pth_path: Union[str, Path],
    onnx_path: Union[str, Path],
    opset_version: int = 18
) -> str:
    """
    Exports a trained PyTorch .pth weight file to .onnx format.
    """
    pth_path = Path(pth_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    base_net = model.convNet()
    if pth_path.exists():
        state_dict = torch.load(pth_path, map_location="cpu")
        base_net.load_state_dict(state_dict)

    wrapper = ExportableCNN(base_net)
    wrapper.eval()

    # Input shape: [batch_size, 3 channels, 80 mel-bins, 15 time-frames]
    dummy_input = torch.randn(1, 3, 80, 15, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["onset_prob"],
        dynamic_shapes={
            "x": {0: torch.export.Dim("batch_size")}
        }
    )

    return str(onnx_path)


class ONNXOnsetDetector:
    """
    Lightweight, portable ONNX Runtime inference engine.
    """
    def __init__(self, onnx_path: Union[str, Path]):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, feats: np.ndarray, minibatch: int = 2048, soundlen: int = 15) -> np.ndarray:
        """
        Feeds spectrogram features through ONNX model.
        feats: numpy array [3, 80, total_frames]
        """
        total_steps = feats.shape[2] - soundlen
        if total_steps <= 0:
            return np.array([])

        results = []
        batch = []
        for i in range(total_steps):
            window = feats[:, :, i:i + soundlen]
            batch.append(window)

            if len(batch) >= minibatch:
                out = self.session.run([self.output_name], {self.input_name: np.array(batch, dtype=np.float32)})[0]
                results.append(out)
                batch = []

        if len(batch) > 0:
            out = self.session.run([self.output_name], {self.input_name: np.array(batch, dtype=np.float32)})[0]
            results.append(out)

        return np.vstack(results).reshape(-1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ODCNN ONNX Exporter")
    parser.add_argument("--model", choices=["don", "ka", "both"], default="both")
    args = parser.parse_args()

    models_dir = Path("./models")
    if args.model in ["don", "both"]:
        don_pth = models_dir / "don_model.pth"
        don_onnx = models_dir / "don_model.onnx"
        if don_pth.exists():
            out = export_to_onnx(don_pth, don_onnx)
            print(f"Exported Don model to {out}")

    if args.model in ["ka", "both"]:
        ka_pth = models_dir / "ka_model.pth"
        ka_onnx = models_dir / "ka_model.onnx"
        if ka_pth.exists():
            out = export_to_onnx(ka_pth, ka_onnx)
            print(f"Exported Ka model to {out}")
