"""
CRNN Architecture for Rhythm Onset Detection.
Combines 2D CNN spectrogram feature extraction, Bidirectional GRU sequence modeling,
and temporal attention pooling for long-range rhythm awareness.
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from music_processor import safe_load_pickle, milden


class RhythmAttention(nn.Module):
    """
    Temporal multi-head rhythm attention pooling layer.
    Allows the network to attend to salient onset patterns across the temporal context window.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(RhythmAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_dim)
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + attn_out)
        return x


class CRNNOnsetMaxNet(nn.Module):
    """
    CRNN Neural Network for Onset Detection:
    - Input: (batch, 3 channels, 80 mel bands, 15 time frames)
    - 2D Conv Layers: extracts spectral-spatial features (spectral energy distribution & harmonics)
    - Frequency Pooling: collapses frequency dimension while preserving temporal sequence
    - Bidirectional GRU: captures temporal evolution, rhythm meter, and note dynamics
    - Multi-Head Attention: pools temporal cues around the center target frame
    - Classifier: maps context to onset probability [0, 1] via Sigmoid
    """

    def __init__(self, in_channels: int = 3, rnn_hidden: int = 64, rnn_layers: int = 2, dropout: float = 0.3):
        super(CRNNOnsetMaxNet, self).__init__()
        self.rnn_hidden = rnn_hidden

        # 2D CNN Frontend
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        # Max pooling across frequency, preserving temporal sequence
        # (3, 80, 15) -> pool1 (2, 1) -> (40, 15) -> pool2 (2, 1) -> (20, 15) -> pool3 (2, 1) -> (10, 15)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # CNN feature map to sequence projection: 64 channels * 10 freq bins = 640
        self.cnn_out_dim = 64 * 10

        # Bidirectional GRU Sequence Model
        self.gru = nn.GRU(
            input_size=self.cnn_out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Rhythm Attention across sequence length (15 frames)
        self.attention = RhythmAttention(hidden_dim=rnn_hidden * 2, num_heads=4)

        # Fully Connected Classifier
        self.fc1 = nn.Linear(rnn_hidden * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, istraining: bool = False, minibatch: int = 1) -> torch.Tensor:
        # x shape: (batch, 3, 80, 15)
        batch_size = x.size(0)

        # CNN feature extraction
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)

        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        # out shape: (batch, 64, 10, 15)

        # Reshape to sequence: (batch, seq_len=15, feature_dim=64*10=640)
        # Permute (batch, channels, freq, time) -> (batch, time, channels, freq) -> flatten
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch_size, out.size(1), -1)

        # Bidirectional GRU
        gru_out, _ = self.gru(out)  # (batch, seq_len, rnn_hidden * 2)

        # Attention enhancement
        attn_out = self.attention(gru_out)  # (batch, seq_len, rnn_hidden * 2)

        # Pool target frame (center frame at index 7 out of 15)
        center_idx = attn_out.size(1) // 2
        center_feat = attn_out[:, center_idx, :]

        # Dense classification
        h = F.relu(self.fc1(center_feat))
        if istraining:
            h = self.dropout(h)
        logits = self.fc2(h)

        return torch.sigmoid(logits)

    def train_data_builder(self, feats, answer, major_note_index, samplerate, soundlen=15, minibatch=1, split=0.2):
        """Builds mini-batches for training."""
        minspace = 0.1
        maxspace = 0.7

        idx = np.random.permutation(major_note_index.shape[0] - soundlen) + soundlen // 2
        X, y = [], []
        cnt = 0

        for i in range(int(idx.shape[0] * split)):
            dist = major_note_index[idx[i] + 1] - major_note_index[idx[i]]
            if minspace * samplerate / 512 < dist < maxspace * samplerate / 512:
                for j in range(-1, dist + 2):
                    start = major_note_index[idx[i]] - soundlen // 2 + j
                    end = major_note_index[idx[i]] + soundlen // 2 + j + 1
                    X.append(feats[:, :, start:end])
                    y.append(answer[major_note_index[idx[i]] + j])
                    cnt += 1

                    if cnt % minibatch == 0:
                        yield (torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float())
                        X, y = [], []

    def infer_data_builder(self, feats, soundlen=15, minibatch=1):
        """Builds mini-batches for inference."""
        x = []
        for i in range(feats.shape[2] - soundlen):
            x.append(feats[:, :, i : i + soundlen])
            if (i + 1) % minibatch == 0:
                yield (torch.from_numpy(np.array(x)).float())
                x = []
        if len(x) != 0:
            yield (torch.from_numpy(np.array(x)).float())

    def train_model(
        self,
        songs,
        minibatch: int = 64,
        epoch: int = 10,
        device: torch.device = torch.device("cpu"),
        soundlen: int = 15,
        save_place: str = "./models/model_crnn.pth",
        log: str = "./data/log/crnn.txt",
        don_ka: int = 0,
        lr: float = 0.001,
    ):
        """Train the CRNN model using Adam optimizer."""
        for song in songs:
            timing = song.timestamp[:, 0]
            sound = song.timestamp[:, 1]
            song.answer = np.zeros((song.feats.shape[2]))

            if don_ka == 0:
                song.major_note_index = np.rint(timing[np.where(sound != 0)] * song.samplerate / 512).astype(np.int32)
            else:
                song.major_note_index = np.rint(timing[np.where(sound == don_ka)] * song.samplerate / 512).astype(np.int32)
                song.minor_note_index = np.rint(timing[np.where(sound == 3 - don_ka)] * song.samplerate / 512).astype(np.int32)

            song.major_note_index = np.delete(song.major_note_index, np.where(song.major_note_index >= song.feats.shape[2]))
            song.minor_note_index = np.delete(song.minor_note_index, np.where(song.minor_note_index >= song.feats.shape[2]))
            song.answer[song.major_note_index] = 1.0
            song.answer[song.minor_note_index] = 0.26
            song.answer = milden(song.answer)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        Path(save_place).parent.mkdir(parents=True, exist_ok=True)
        Path(log).parent.mkdir(parents=True, exist_ok=True)

        for i in range(epoch):
            running_loss = 0.0
            for song in songs:
                for X, y in self.train_data_builder(
                    song.feats, song.answer, song.major_note_index, song.samplerate, soundlen, minibatch, split=0.2
                ):
                    optimizer.zero_grad()
                    output = self(X.to(device), istraining=True, minibatch=minibatch)
                    loss = criterion(output.squeeze(), y.to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            with open(log, "a") as f:
                print(f"epoch: {i+1} running_loss: {running_loss:.10f}", file=f)
            print(f"epoch: {i+1} running_loss: {running_loss:.10f}")

        torch.save(self.state_dict(), save_place)
        print(f"Model saved to {save_place}")

    def infer(self, feats, device: torch.device, minibatch: int = 128) -> np.ndarray:
        """Run forward onset inference across entire song feature matrix."""
        self.eval()
        with torch.no_grad():
            inference = []
            for x in tqdm(self.infer_data_builder(feats, minibatch=minibatch), total=max(1, feats.shape[2] // minibatch)):
                output = self(x.to(device), istraining=False, minibatch=x.shape[0])
                inference.append(output.cpu().numpy().reshape(-1))

            if len(inference) == 0:
                return np.array([])
            return np.concatenate(inference).reshape(-1)


def initialize_default_crnn_weights():
    """Generates initial baseline weights for CRNN Don and Ka models if not present."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    don_path = models_dir / "don_crnn.pth"
    ka_path = models_dir / "ka_crnn.pth"

    if not don_path.exists():
        model = CRNNOnsetMaxNet()
        torch.save(model.state_dict(), don_path)
        print(f"Initialized default CRNN Don weights at {don_path}")

    if not ka_path.exists():
        model = CRNNOnsetMaxNet()
        torch.save(model.state_dict(), ka_path)
        print(f"Initialized default CRNN Ka weights at {ka_path}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CRNN Module Running on device: {device}")
    initialize_default_crnn_weights()
