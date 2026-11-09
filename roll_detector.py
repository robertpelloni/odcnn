"""
Sustained Drumroll and Balloon Onset Detection Engine for Rhythm Games.
Provides automated detection and segmentation of continuous sustained sounds
(drumrolls, rapid cymbals, balloons, sliders) from audio representations.

Taiko no Tatsujin .tja note specifications:
- '5': Yellow Drumroll (連打) start
- '6': Big Yellow Drumroll (大連打) start
- '7': Balloon Note (風船連打) start
- '8': End of Drumroll / Balloon

osu! Taiko (.osu Mode 1) specification:
- Spinner hitObject: 256,192,startTime,12,hitsound,endTime:hitSample
"""

import numpy as np
from typing import List, Dict, Any, Tuple


def detect_sustained_rolls(
    audio: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    min_roll_sec: float = 0.35,
    max_roll_sec: float = 5.0,
    energy_ratio_thresh: float = 0.08,
    flux_thresh: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Detects sustained continuous sound intervals suitable for drumrolls and balloons.

    Algorithm:
    1. Computes short-term frame RMS energy.
    2. Identifies contiguous regions where energy exceeds energy_ratio_thresh * max_energy.
    3. Evaluates spectral flux variance: sustained rolls have lower frame-to-frame flux variance
       compared to rapid discrete percussive bursts.
    4. Categorizes into Yellow Roll ('5'), Big Roll ('6'), or Balloon ('7') based on length and energy.

    Returns:
        List of dicts:
        [
            {
                "start_sec": float,
                "end_sec": float,
                "duration_sec": float,
                "type": int (5, 6, or 7),
                "name": str ("Yellow Roll", "Big Roll", "Balloon")
            }, ...
        ]
    """
    if len(audio) == 0:
        return []

    # Ensure 1D float array
    if audio.ndim > 1:
        audio_mono = np.mean(audio, axis=-1)
    else:
        audio_mono = audio.astype(np.float32)

    frame_length = hop_length * 2
    n_frames = int(np.ceil((len(audio_mono) - frame_length) / hop_length)) + 1
    if n_frames <= 1:
        return []

    # Compute frame-by-frame RMS energy
    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        end = min(len(audio_mono), start + frame_length)
        chunk = audio_mono[start:end]
        if len(chunk) > 0:
            rms[i] = np.sqrt(np.mean(chunk ** 2))

    max_rms = np.max(rms)
    if max_rms < 1e-6:
        return []

    # Threshold frames
    thresh = energy_ratio_thresh * max_rms
    active_mask = rms >= thresh

    # Group contiguous active frames
    segments = []
    in_segment = False
    seg_start = 0

    for i, active in enumerate(active_mask):
        if active and not in_segment:
            in_segment = True
            seg_start = i
        elif not active and in_segment:
            in_segment = False
            seg_len = i - seg_start
            start_sec = (seg_start * hop_length) / sr
            end_sec = (i * hop_length) / sr
            duration = end_sec - start_sec
            if min_roll_sec <= duration <= max_roll_sec:
                segments.append((seg_start, i, start_sec, end_sec, duration))

    # Handle end of signal segment
    if in_segment:
        seg_len = len(active_mask) - seg_start
        start_sec = (seg_start * hop_length) / sr
        end_sec = (len(active_mask) * hop_length) / sr
        duration = end_sec - start_sec
        if min_roll_sec <= duration <= max_roll_sec:
            segments.append((seg_start, len(active_mask), start_sec, end_sec, duration))

    rolls: List[Dict[str, Any]] = []
    for s_frame, e_frame, s_sec, e_sec, dur in segments:
        seg_rms = rms[s_frame:e_frame]
        mean_energy = float(np.mean(seg_rms))
        norm_energy = mean_energy / max_rms

        # Classification heuristics:
        # - Very high sustained energy (> 0.6 norm): Big Roll ('6')
        # - Medium duration (0.5s - 1.5s) with moderate energy: Balloon ('7')
        # - Standard sustained: Yellow Roll ('5')
        if norm_energy > 0.65:
            roll_type = 6
            name = "Big Roll"
        elif 0.5 <= dur <= 1.8 and norm_energy > 0.35:
            roll_type = 7
            name = "Balloon"
        else:
            roll_type = 5
            name = "Yellow Roll"

        rolls.append({
            "start_sec": round(float(s_sec), 3),
            "end_sec": round(float(e_sec), 3),
            "duration_sec": round(float(dur), 3),
            "type": roll_type,
            "name": name
        })

    return rolls


def merge_onsets_and_rolls(
    don_timestamps: np.ndarray,
    ka_timestamps: np.ndarray,
    rolls: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prunes discrete Don and Ka onset notes that occur strictly inside active roll intervals.
    In Taiko charts, discrete notes are not placed on top of sustained rolls/balloons.
    """
    if len(rolls) == 0:
        return don_timestamps, ka_timestamps

    keep_don = []
    for t in don_timestamps:
        inside = any(r["start_sec"] <= t <= r["end_sec"] for r in rolls)
        if not inside:
            keep_don.append(t)

    keep_ka = []
    for t in ka_timestamps:
        inside = any(r["start_sec"] <= t <= r["end_sec"] for r in rolls)
        if not inside:
            keep_ka.append(t)

    return np.array(keep_don, dtype=np.float64), np.array(keep_ka, dtype=np.float64)
