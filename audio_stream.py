"""
Audio Stream Processor and Streaming Chunk Simulator.
Enables real-time chunk-by-chunk onset detection and streaming simulation
for physical or synthetic input streams without requiring full offline tracks.
"""

import numpy as np
import librosa
from typing import Generator, List, Dict, Tuple, Optional


class AudioChunkStreamer:
    """
    Simulates a streaming audio source by chunking a loaded audio signal or buffer.
    """

    def __init__(self, audio_data: np.ndarray, sample_rate: int = 44100, chunk_size: int = 1024):
        self.audio_data = audio_data.astype(np.float32)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.total_samples = len(audio_data)

    @classmethod
    def from_file(cls, audio_path: str, sample_rate: int = 44100, chunk_size: int = 1024):
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        return cls(y, sample_rate=sr, chunk_size=chunk_size)

    def stream_chunks(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (chunk_index, timestamp_sec, chunk_samples)
        """
        for i in range(0, self.total_samples, self.chunk_size):
            chunk = self.audio_data[i : i + self.chunk_size]
            timestamp = i / float(self.sample_rate)
            yield (i // self.chunk_size, timestamp, chunk)


class StreamingOnsetDetector:
    """
    Streaming buffer onset detector for live or chunked audio.
    Maintains a rolling ring buffer of audio frames and computes local spectral energy & onsets.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_seconds: float = 1.0,
        energy_threshold: float = 0.05,
        refractory_period_sec: float = 0.08
    ):
        self.sample_rate = sample_rate
        self.buffer_size = int(buffer_seconds * sample_rate)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.energy_threshold = energy_threshold
        self.refractory_samples = int(refractory_period_sec * sample_rate)
        self.samples_since_last_onset = self.refractory_samples
        self.detected_onsets: List[Dict[str, float]] = []
        self.total_processed_samples = 0

    def push_chunk(self, chunk: np.ndarray) -> List[Dict[str, float]]:
        """
        Push incoming audio chunk into buffer and detect any onsets within this chunk.
        Returns newly triggered onsets: [{'timestamp': float, 'type': 'don'|'ka', 'confidence': float}]
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        chunk_len = len(chunk)
        if chunk_len == 0:
            return []

        # Shift ring buffer
        if chunk_len >= self.buffer_size:
            self.buffer = chunk[-self.buffer_size:].copy()
        else:
            self.buffer[:-chunk_len] = self.buffer[chunk_len:]
            self.buffer[-chunk_len:] = chunk

        new_onsets = []

        # Calculate instantaneous RMS energy of the chunk
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-9))
        self.total_processed_samples += chunk_len
        self.samples_since_last_onset += chunk_len

        # Detection logic: RMS exceeding threshold and refractory period elapsed
        if rms > self.energy_threshold and self.samples_since_last_onset >= self.refractory_samples:
            # Distinguish Don (low-frequency dominance) vs Ka (high-frequency dominance) via spectral centroid
            # Compute lightweight zero crossing rate or FFT spectral centroid on chunk
            fft_mag = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.sample_rate)
            centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))

            onset_type = "don" if centroid < 2200.0 else "ka"
            timestamp = (self.total_processed_samples - chunk_len / 2.0) / float(self.sample_rate)

            onset_record = {
                "timestamp": round(timestamp, 4),
                "type": onset_type,
                "confidence": round(min(1.0, rms * 5.0), 3),
                "centroid_hz": round(centroid, 1)
            }
            new_onsets.append(onset_record)
            self.detected_onsets.append(onset_record)
            self.samples_since_last_onset = 0

        return new_onsets

    def reset(self):
        self.buffer.fill(0)
        self.samples_since_last_onset = self.refractory_samples
        self.detected_onsets.clear()
        self.total_processed_samples = 0
