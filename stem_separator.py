"""
Audio Stem Separation Engine (Harmonic/Percussive & Sub-Band Spectral Filtering)
================================================================================
Provides zero-dependency audio stem separation utilizing Median-Filtered
Harmonic-Percussive Source Separation (HPSS) on Short-Time Fourier Transforms (STFT),
as well as sub-band frequency filtering (Low/Bass, Mid, High/Treble).

Isolating percussive transients from sustained harmonic tones drastically improves
transient onset picking accuracy for Don (kick) and Ka (snare/rim) notes in noisy mixes.
"""

import os
import math
import numpy as np
import scipy.signal
import soundfile as sf


def separate_hpss(audio_data: np.ndarray, sr: int = 44100, n_fft: int = 2048,
                  hop_length: int = 512, kernel_size: int = 31, power: float = 2.0):
    """
    Separates a 1D audio waveform into Harmonic and Percussive components.

    Parameters:
        audio_data: 1D numpy array of audio samples (float32).
        sr: Sampling rate (e.g., 44100 Hz).
        n_fft: FFT window size.
        hop_length: STFT hop length.
        kernel_size: Median filter kernel size (must be odd).
        power: Exponent for soft masking.

    Returns:
        harmonic_audio: 1D numpy array of harmonic audio samples.
        percussive_audio: 1D numpy array of percussive audio samples.
        h_ratio: Float ratio of harmonic energy to total energy.
        p_ratio: Float ratio of percussive energy to total energy.
    """
    if audio_data.ndim > 1:
        # Mixdown to mono if multichannel
        audio_data = np.mean(audio_data, axis=1)

    audio_len = len(audio_data)
    if audio_len == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 0.0, 0.0

    # Window function
    window = scipy.signal.windows.hann(n_fft, sym=False)

    # Compute complex STFT using scipy.signal.stft
    f, t, stft_matrix = scipy.signal.stft(
        audio_data,
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary='zeros',
        padded=True
    )

    magnitude = np.abs(stft_matrix)

    # Median filtering:
    # Harmonic components are horizontal lines across time (filter along time axis, axis=1)
    # Percussive components are vertical lines across frequency (filter along frequency axis, axis=0)
    k_time = kernel_size
    k_freq = kernel_size
    if k_time % 2 == 0:
        k_time += 1
    if k_freq % 2 == 0:
        k_freq += 1

    harm_mag = scipy.signal.medfilt2d(magnitude, [1, k_time])
    perc_mag = scipy.signal.medfilt2d(magnitude, [k_freq, 1])

    # Soft masking
    h_pow = harm_mag ** power
    p_pow = perc_mag ** power
    total_pow = h_pow + p_pow + 1e-10

    mask_h = h_pow / total_pow
    mask_p = p_pow / total_pow

    stft_h = mask_h * stft_matrix
    stft_p = mask_p * stft_matrix

    # Inverse STFT to reconstruct time-domain waveforms
    _, harm_audio = scipy.signal.istft(
        stft_h,
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=True
    )

    _, perc_audio = scipy.signal.istft(
        stft_p,
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=True
    )

    # Match original length
    harm_audio = harm_audio[:audio_len].astype(np.float32)
    perc_audio = perc_audio[:audio_len].astype(np.float32)

    # Calculate energy distribution
    harm_energy = float(np.sum(harm_audio ** 2))
    perc_energy = float(np.sum(perc_audio ** 2))
    total_energy = harm_energy + perc_energy + 1e-12

    h_ratio = round(harm_energy / total_energy, 4)
    p_ratio = round(perc_energy / total_energy, 4)

    return harm_audio, perc_audio, h_ratio, p_ratio


def separate_subbands(audio_data: np.ndarray, sr: int = 44100,
                      low_cutoff: float = 250.0, high_cutoff: float = 2500.0):
    """
    Separates an audio signal into 3 sub-band frequency stems:
    - Bass/Low: < low_cutoff (e.g. kick drums, bassline)
    - Mid: low_cutoff to high_cutoff (vocals, snare bodies)
    - High/Treble: > high_cutoff (cymbals, hi-hats, rimshots)
    """
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    nyquist = 0.5 * sr
    low_norm = min(max(low_cutoff / nyquist, 0.001), 0.99)
    high_norm = min(max(high_cutoff / nyquist, 0.001), 0.99)

    # Low-pass filter (order 4 Butterworth)
    b_low, a_low = scipy.signal.butter(4, low_norm, btype='low')
    bass = scipy.signal.filtfilt(b_low, a_low, audio_data).astype(np.float32)

    # Band-pass filter for Mid
    b_mid, a_mid = scipy.signal.butter(4, [low_norm, high_norm], btype='band')
    mid = scipy.signal.filtfilt(b_mid, a_mid, audio_data).astype(np.float32)

    # High-pass filter for Treble
    b_high, a_high = scipy.signal.butter(4, high_norm, btype='high')
    treble = scipy.signal.filtfilt(b_high, a_high, audio_data).astype(np.float32)

    return bass, mid, treble


def extract_and_save_stems(audio_file_path: str, output_dir: str = "data/stems",
                           sr: int = 44100) -> dict:
    """
    Reads an audio file, applies HPSS separation and 3-band filtering,
    and writes separated stems to output_dir:
    - percussive.wav
    - harmonic.wav
    - bass.wav
    - treble.wav

    Returns stem metadata dictionary with energy metrics and file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_data, file_sr = sf.read(audio_file_path, dtype='float32')

    if file_sr != sr:
        # Resample if needed
        num_samples = int(len(audio_data) * float(sr) / file_sr)
        audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)

    harm_audio, perc_audio, h_ratio, p_ratio = separate_hpss(audio_data, sr=sr)
    bass_audio, mid_audio, treble_audio = separate_subbands(audio_data, sr=sr)

    perc_path = os.path.join(output_dir, "stem_percussive.wav")
    harm_path = os.path.join(output_dir, "stem_harmonic.wav")
    bass_path = os.path.join(output_dir, "stem_bass.wav")
    treble_path = os.path.join(output_dir, "stem_treble.wav")

    sf.write(perc_path, perc_audio, sr)
    sf.write(harm_path, harm_audio, sr)
    sf.write(bass_path, bass_audio, sr)
    sf.write(treble_path, treble_audio, sr)

    duration_sec = round(len(audio_data) / float(sr), 2)

    return {
        "status": "success",
        "duration_sec": duration_sec,
        "harmonic_ratio": h_ratio,
        "percussive_ratio": p_ratio,
        "stems": {
            "percussive": perc_path,
            "harmonic": harm_path,
            "bass": bass_path,
            "treble": treble_path,
        }
    }
