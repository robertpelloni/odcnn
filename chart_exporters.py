"""
Multi-format Rhythm Game Chart Exporters.
Provides automated export from ODCNN detected onsets into:
1. osu! Taiko (.osu Mode 1)
2. StepMania (.sm)
3. Taiko no Tatsujin (.tja)
"""

import numpy as np
from pathlib import Path
from typing import Optional, List

def export_osu(
    filename: str,
    song,
    don_timestamp: np.ndarray,
    ka_timestamp: np.ndarray,
    title: str = "ODCNN Inferred Track",
    artist: str = "ODCNN AI",
    bpm: float = 120.0
) -> str:
    """
    Exports detected onsets to osu! Taiko format (.osu Mode: 1).
    Don = Normal hitsound (0)
    Ka = Whistle/Clap hitsound (2 / 8)
    """
    beat_length = 60000.0 / bpm

    don_ms = np.rint(don_timestamp * 512 / song.samplerate * 1000).astype(np.int64)
    ka_ms = np.rint(ka_timestamp * 512 / song.samplerate * 1000).astype(np.int64)

    # Combine notes with type (don=0, ka=2)
    notes = []
    for t in don_ms:
        notes.append((t, 0))  # Don hitsound
    for t in ka_ms:
        notes.append((t, 2))  # Ka hitsound (whistle)

    notes.sort(key=lambda x: x[0])

    lines = [
        "osu file format v14",
        "",
        "[General]",
        "AudioFilename: audio.wav",
        "AudioLeadIn: 0",
        "PreviewTime: 10000",
        "Countdown: 0",
        "SampleSet: Normal",
        "StackLeniency: 0.7",
        "Mode: 1",  # 1 = osu!taiko
        "LetterboxInBreaks: 0",
        "WidescreenStoryboard: 0",
        "",
        "[Metadata]",
        f"Title:{title}",
        f"TitleUnicode:{title}",
        f"Artist:{artist}",
        f"ArtistUnicode:{artist}",
        "Creator:ODCNN Autopilot",
        "Version:AI Inferred Onsets",
        "Source:ODCNN",
        "Tags:odcnn taiko onset neural network",
        "",
        "[Difficulty]",
        "HPDrainRate:6",
        "CircleSize:5",
        "OverallDifficulty:6",
        "ApproachRate:6",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[TimingPoints]",
        f"0,{beat_length:.6f},4,1,0,100,1,0",
        "",
        "[HitObjects]"
    ]

    for time_ms, hitsound in notes:
        # Format: x,y,time,type,hitSound,objectParams,hitSample
        # Taiko standard coordinates: x=256, y=192, type=1 (circle)
        lines.append(f"256,192,{time_ms},1,{hitsound},0:0:0:0:")

    content = "\n".join(lines) + "\n"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename


def export_sm(
    filename: str,
    song,
    don_timestamp: np.ndarray,
    ka_timestamp: np.ndarray,
    title: str = "ODCNN Inferred Track",
    artist: str = "ODCNN AI",
    bpm: float = 120.0
) -> str:
    """
    Exports detected onsets to StepMania (.sm) format.
    Notes are quantized to 16th-note measure grids.
    Don = Left/Right (1000, 0001)
    Ka = Up/Down (0100, 0010)
    """
    sec_per_beat = 60.0 / bpm
    subdivisions_per_measure = 16  # 16th notes
    sec_per_step = sec_per_beat / 4.0

    don_sec = don_timestamp * 512 / song.samplerate
    ka_sec = ka_timestamp * 512 / song.samplerate

    max_time = 0.0
    if len(don_sec) > 0:
        max_time = max(max_time, np.max(don_sec))
    if len(ka_sec) > 0:
        max_time = max(max_time, np.max(ka_sec))

    total_steps = int(np.ceil(max_time / sec_per_step)) + subdivisions_per_measure
    total_measures = int(np.ceil(total_steps / subdivisions_per_measure))

    # Grid of measures, each containing 16 steps of 4-character rows (0000)
    measures = [["0000" for _ in range(subdivisions_per_measure)] for _ in range(total_measures)]

    for t in don_sec:
        step_idx = int(round(t / sec_per_step))
        m_idx = step_idx // subdivisions_per_measure
        s_idx = step_idx % subdivisions_per_measure
        if m_idx < total_measures:
            measures[m_idx][s_idx] = "1001"  # Don: Left + Right

    for t in ka_sec:
        step_idx = int(round(t / sec_per_step))
        m_idx = step_idx // subdivisions_per_measure
        s_idx = step_idx % subdivisions_per_measure
        if m_idx < total_measures:
            # If already has Don, override or alternate
            measures[m_idx][s_idx] = "0110"  # Ka: Up + Down

    sm_header = [
        f"#TITLE:{title};",
        "#SUBTITLE:AI Inferred;",
        f"#ARTIST:{artist};",
        "#GENRE:ODCNN Rhythm;",
        "#CREDIT:ODCNN Autopilot;",
        "#MUSIC:audio.wav;",
        "#OFFSET:0.000000;",
        "#SAMPLESTART:10.000000;",
        "#SAMPLELENGTH:12.000000;",
        "#SELECTABLE:YES;",
        f"#BPMS:0.000={bpm:.3f};",
        "#STOPS:;",
        "",
        "//---------------dance-single - ODCNN AI---------------",
        "#NOTES:",
        "     dance-single:",
        "     ODCNN Autopilot:",
        "     Challenge:",
        "     10:",
        "     0.100,0.200,0.300,0.400,0.500:"
    ]

    measure_strings = []
    for m in measures:
        measure_strings.append("\n".join(m))

    notes_body = "\n,\n".join(measure_strings) + "\n;\n"
    content = "\n".join(sm_header) + "\n" + notes_body

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename
