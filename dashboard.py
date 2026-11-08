#!/usr/bin/env python3
"""
ODCNN Unified Autopilot Dashboard & Control Center.
Consolidates all music processing, CNN training, inference, peak detection,
and Taiko chart/audio synthesis into a single high-value responsive web interface.
"""

import os
import sys
import time
import json
import psutil
import shutil
import pickle
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf
import librosa
import torch
from fastapi import FastAPI, Request, BackgroundTasks, Form, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import model
import music_processor
import synthesize
import export_onnx
import audio_stream
import chart_exporters

# Global paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PICKLES_DIR = DATA_DIR / "pickles"
INFERENCE_DIR = DATA_DIR / "inference"
LOGS_DIR = DATA_DIR / "log"
STATIC_DIR = BASE_DIR / "static"

for directory in [DATA_DIR, MODELS_DIR, PICKLES_DIR, INFERENCE_DIR, LOGS_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Application state
app = FastAPI(title="ODCNN Autopilot Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory logging ring buffer & concurrency locks
SERVER_LOGS = []
LOG_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()

def log_event(message: str):
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with LOG_LOCK:
        SERVER_LOGS.append(formatted)
        if len(SERVER_LOGS) > 200:
            SERVER_LOGS.pop(0)

# Background job tracker
CURRENT_JOB = {
    "status": "idle",
    "job_name": None,
    "progress": 0,
    "message": "Ready"
}

def run_background_task(name: str, target_func, *args, **kwargs):
    def wrapper():
        global CURRENT_JOB
        CURRENT_JOB["status"] = "running"
        CURRENT_JOB["job_name"] = name
        CURRENT_JOB["progress"] = 0
        CURRENT_JOB["message"] = f"Started {name}"
        log_event(f"JOB STARTED: {name}")
        try:
            target_func(*args, **kwargs)
            CURRENT_JOB["status"] = "completed"
            CURRENT_JOB["progress"] = 100
            CURRENT_JOB["message"] = f"Finished {name} successfully"
            log_event(f"JOB COMPLETED: {name}")
        except Exception as e:
            CURRENT_JOB["status"] = "error"
            CURRENT_JOB["message"] = f"Error in {name}: {str(e)}"
            log_event(f"JOB ERROR in {name}: {str(e)}")
        finally:
            time.sleep(3)
            if CURRENT_JOB["status"] == "completed":
                CURRENT_JOB["status"] = "idle"
                CURRENT_JOB["job_name"] = None
                CURRENT_JOB["message"] = "Ready"

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()

# API Endpoints
@app.get("/api/status")
def get_status():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Engine"

    # Inspect available files
    has_test_pickle = (PICKLES_DIR / "test_data.pickle").exists()
    has_train_pickle = (PICKLES_DIR / "train_reduced.pickle").exists() or (PICKLES_DIR / "train_data.pickle").exists()
    has_don_model = (MODELS_DIR / "don_model.pth").exists()
    has_ka_model = (MODELS_DIR / "ka_model.pth").exists()
    has_don_onnx = (MODELS_DIR / "don_model.onnx").exists()
    has_ka_onnx = (MODELS_DIR / "ka_model.onnx").exists()
    has_don_infer = (PICKLES_DIR / "don_inference.pickle").exists()
    has_ka_infer = (PICKLES_DIR / "ka_inference.pickle").exists()
    has_synthesized_wav = (INFERENCE_DIR / "inferred_music.wav").exists()
    has_synthesized_tja = (INFERENCE_DIR / "inferred_notes.tja").exists()
    has_synthesized_osu = (INFERENCE_DIR / "inferred_notes.osu").exists()
    has_synthesized_sm = (INFERENCE_DIR / "inferred_notes.sm").exists()
    has_synthesized_dat = (INFERENCE_DIR / "inferred_notes.dat").exists()

    with open(BASE_DIR / "VERSION.md", "r") as vf:
        version_str = vf.read().strip()

    return {
        "version": version_str,
        "cpu_usage": cpu,
        "ram_usage": mem.percent,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "job": CURRENT_JOB,
        "artifacts": {
            "test_pickle": has_test_pickle,
            "train_pickle": has_train_pickle,
            "don_model": has_don_model,
            "ka_model": has_ka_model,
            "don_onnx": has_don_onnx,
            "ka_onnx": has_ka_onnx,
            "don_inference": has_don_infer,
            "ka_inference": has_ka_infer,
            "synthesized_wav": has_synthesized_wav,
            "synthesized_tja": has_synthesized_tja,
            "synthesized_osu": has_synthesized_osu,
            "synthesized_sm": has_synthesized_sm,
            "synthesized_dat": has_synthesized_dat,
        }
    }

@app.get("/api/logs")
def get_logs():
    with LOG_LOCK:
        return {"logs": list(SERVER_LOGS)}

@app.post("/api/action/preprocess")
def action_preprocess(dataset: str = Form("test")):
    if CURRENT_JOB["status"] == "running":
        return JSONResponse({"error": "Another job is already running"}, status_code=400)

    def do_preprocess():
        log_event(f"Starting audio preprocessing for '{dataset}'...")
        if dataset == "test":
            music_processor.music_for_test("./data/test/")
        elif dataset == "reduced":
            music_processor.music_for_train_reduced("./data/train_reduced/*", verbose=True, difficulty=0, diff=True)
        elif dataset == "val":
            music_processor.music_for_validation("./data/validation")
        elif dataset == "train":
            music_processor.music_for_train("./data/train/*", verbose=True, difficulty=0, diff=True)
        else:
            raise ValueError(f"Unknown dataset mode: {dataset}")

    run_background_task(f"Preprocess ({dataset})", do_preprocess)
    return {"status": "ok", "message": f"Preprocessing for {dataset} initiated"}

@app.post("/api/action/train")
def action_train(model_type: str = Form("don"), epochs: int = Form(10), batch_size: int = Form(64)):
    if CURRENT_JOB["status"] == "running":
        return JSONResponse({"error": "Another job is already running"}, status_code=400)

    def do_train():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = model.convNet().to(device)

        pickle_path = PICKLES_DIR / "train_data.pickle"
        if not pickle_path.exists():
            pickle_path = PICKLES_DIR / "train_reduced.pickle"

        log_event(f"Loading training data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            songs = pickle.load(f)

        save_pth = str(MODELS_DIR / f"{model_type}_model.pth")
        log_path = str(LOGS_DIR / f"{model_type}.txt")
        don_ka_flag = 1 if model_type == "don" else 2

        log_event(f"Training {model_type.upper()} model for {epochs} epochs (batch size: {batch_size})...")
        net.train(
            songs=songs,
            minibatch=batch_size,
            epoch=epochs,
            device=device,
            soundlen=15,
            save_place=save_pth,
            log=log_path,
            don_ka=don_ka_flag
        )
        log_event(f"Model saved to {save_pth}")

    run_background_task(f"Train {model_type.upper()} Model", do_train)
    return {"status": "ok", "message": f"Training {model_type} started"}

@app.post("/api/action/infer")
def action_infer(model_type: str = Form("both"), engine: str = Form("pytorch"), arch: str = Form("cnn")):
    if CURRENT_JOB["status"] == "running":
        return JSONResponse({"error": "Another job is already running"}, status_code=400)

    def do_infer():
        log_event(f"Loading test song pickle (arch={arch.upper()}, engine={engine.upper()})...")
        song = music_processor.safe_load_pickle(PICKLES_DIR / "test_data.pickle")

        types_to_run = ["don", "ka"] if model_type == "both" else [model_type]
        for t in types_to_run:
            if engine == "onnx":
                onnx_name = f"{t}_crnn.onnx" if arch == "crnn" else f"{t}_model.onnx"
                onnx_file = MODELS_DIR / onnx_name
                pth_file = MODELS_DIR / (f"{t}_crnn.pth" if arch == "crnn" else f"{t}_model.pth")
                if not onnx_file.exists():
                    log_event(f"Exporting {t.upper()} ({arch.upper()}) weights to ONNX first...")
                    export_onnx.export_to_onnx(pth_file, onnx_file, arch=arch)

                log_event(f"Running ONNX Runtime inference for {t.upper()} ({arch.upper()})...")
                detector = export_onnx.ONNXOnsetDetector(onnx_file)
                inference = detector.infer(song.feats, minibatch=2048)
            else:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                if arch == "crnn":
                    import model_crnn
                    net = model_crnn.CRNNOnsetMaxNet().to(device)
                    pth = MODELS_DIR / f"{t}_crnn.pth"
                    if not pth.exists():
                        model_crnn.initialize_default_crnn_weights()
                else:
                    net = model.convNet().to(device)
                    pth = MODELS_DIR / f"{t}_model.pth"

                log_event(f"Loading {arch.upper()} weights from {pth}...")
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load(pth))
                else:
                    net.load_state_dict(torch.load(pth, map_location="cpu"))

                log_event(f"Running PyTorch inference for {t.upper()} ({arch.upper()})...")
                inference = net.infer(song.feats, device, minibatch=2048)
                inference = np.reshape(inference, (-1))

            out_pickle = PICKLES_DIR / f"{t}_inference.pickle"
            with open(out_pickle, "wb") as f:
                pickle.dump(inference, f)
            log_event(f"Saved inference to {out_pickle}")

    run_background_task(f"Inference ({model_type}/{arch}/{engine})", do_infer)
    return {"status": "ok", "message": f"Inference for {model_type} ({arch}/{engine}) started"}

@app.post("/api/action/export_onnx")
def action_export_onnx(target: str = Form("both")):
    if CURRENT_JOB["status"] == "running":
        return JSONResponse({"error": "Another job is already running"}, status_code=400)

    def do_export():
        targets = ["don", "ka"] if target == "both" else [target]
        for t in targets:
            pth = MODELS_DIR / f"{t}_model.pth"
            onnx_path = MODELS_DIR / f"{t}_model.onnx"
            log_event(f"Exporting {pth} to {onnx_path}...")
            export_onnx.export_to_onnx(pth, onnx_path)
            log_event(f"ONNX graph exported for {t.upper()}")

    run_background_task(f"Export ONNX ({target})", do_export)
    return {"status": "ok", "message": f"ONNX export initiated for {target}"}

@app.post("/api/action/synthesize")
def action_synthesize(
    pre_max: int = Form(1),
    post_max: int = Form(2),
    pre_avg: int = Form(4),
    post_avg: int = Form(5),
    delta: float = Form(0.05),
    wait: int = Form(3)
):
    if CURRENT_JOB["status"] == "running":
        return JSONResponse({"error": "Another job is already running"}, status_code=400)

    def do_synthesize():
        log_event("Loading pickles for synthesis...")
        with open(PICKLES_DIR / "test_data.pickle", "rb") as f:
            song = pickle.load(f)
        with open(PICKLES_DIR / "don_inference.pickle", "rb") as f:
            don_inference = pickle.load(f)
        with open(PICKLES_DIR / "ka_inference.pickle", "rb") as f:
            ka_inference = pickle.load(f)

        log_event(f"Running peak detection (delta={delta}, wait={wait})...")
        don_inference_smooth = synthesize.smooth(don_inference, 5)
        ka_inference_smooth = synthesize.smooth(ka_inference, 5)

        don_timestamp = (synthesize.peak_pick(
            don_inference_smooth,
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=delta,
            wait=wait
        ) + 7)

        ka_timestamp = (synthesize.peak_pick(
            ka_inference_smooth,
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=delta,
            wait=wait
        ) + 7)

        log_event(f"Detected {len(don_timestamp)} raw Don peaks, {len(ka_timestamp)} raw Ka peaks")

        song.don_timestamp = don_timestamp[np.where(don_inference_smooth[don_timestamp] > ka_inference_smooth[don_timestamp])]
        song.timestamp = song.don_timestamp * 512 / song.samplerate
        song.synthesize(diff='don')

        song.ka_timestamp = ka_timestamp[np.where(ka_inference_smooth[ka_timestamp] > don_inference_smooth[ka_timestamp])]
        song.timestamp = song.ka_timestamp * 512 / song.samplerate
        song.synthesize(diff='ka')

        out_wav = INFERENCE_DIR / "inferred_music.wav"
        song.save(str(out_wav))
        log_event(f"Synthesized audio written to {out_wav}")

        out_tja = INFERENCE_DIR / "inferred_notes.tja"
        synthesize.create_tja(str(out_tja), song, song.don_timestamp, song.ka_timestamp)
        log_event(f"Taiko chart (.tja) written to {out_tja}")

        out_osu = INFERENCE_DIR / "inferred_notes.osu"
        synthesize.export_osu(str(out_osu), song, song.don_timestamp, song.ka_timestamp)
        log_event(f"osu! Taiko chart (.osu) written to {out_osu}")

        out_sm = INFERENCE_DIR / "inferred_notes.sm"
        synthesize.export_sm(str(out_sm), song, song.don_timestamp, song.ka_timestamp)
        log_event(f"StepMania chart (.sm) written to {out_sm}")

        out_dat = INFERENCE_DIR / "inferred_notes.dat"
        synthesize.export_beat_saber(str(out_dat), song, song.don_timestamp, song.ka_timestamp)
        log_event(f"Beat Saber chart (.dat) written to {out_dat}")

        out_mid = INFERENCE_DIR / "inferred_notes.mid"
        synthesize.export_midi(str(out_mid), song, song.don_timestamp, song.ka_timestamp)
        log_event(f"Standard MIDI chart (.mid) written to {out_mid}")

    run_background_task("Synthesize Chart & Audio", do_synthesize)
    return {"status": "ok", "message": "Synthesis initiated"}

@app.post("/api/action/quit")
def action_quit():
    log_event("Server shutdown requested via API")
    def shutdown():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=shutdown, daemon=True).start()
    return {"status": "ok", "message": "Server shutting down..."}

@app.get("/api/download/audio")
def download_audio():
    path = INFERENCE_DIR / "inferred_music.wav"
    if path.exists():
        return FileResponse(path, media_type="audio/wav", filename="inferred_music.wav")
    return JSONResponse({"error": "Audio file not found"}, status_code=404)

@app.get("/api/download/tja")
def download_tja():
    path = INFERENCE_DIR / "inferred_notes.tja"
    if path.exists():
        return FileResponse(path, media_type="text/plain", filename="inferred_notes.tja")
    return JSONResponse({"error": "Chart file not found"}, status_code=404)

@app.get("/api/download/osu")
def download_osu():
    path = INFERENCE_DIR / "inferred_notes.osu"
    if path.exists():
        return FileResponse(path, media_type="text/plain", filename="inferred_notes.osu")
    return JSONResponse({"error": "osu! chart file not found"}, status_code=404)

@app.get("/api/download/sm")
def download_sm():
    path = INFERENCE_DIR / "inferred_notes.sm"
    if path.exists():
        return FileResponse(path, media_type="text/plain", filename="inferred_notes.sm")
    return JSONResponse({"error": "StepMania chart file not found"}, status_code=404)

@app.get("/api/download/beatsaber")
def download_beatsaber():
    path = INFERENCE_DIR / "inferred_notes.dat"
    if path.exists():
        return FileResponse(path, media_type="application/json", filename="inferred_notes.dat")
    return JSONResponse({"error": "Beat Saber chart file not found"}, status_code=404)

@app.get("/api/download/midi")
def download_midi():
    path = INFERENCE_DIR / "inferred_notes.mid"
    if path.exists():
        return FileResponse(path, media_type="audio/midi", filename="inferred_notes.mid")
    return JSONResponse({"error": "MIDI chart file not found"}, status_code=404)

@app.get("/api/download/sample_music")
def download_sample_music():
    path = DATA_DIR / "test" / "test_music.ogg"
    if path.exists():
        return FileResponse(path, media_type="audio/ogg", filename="test_music.ogg")
    return JSONResponse({"error": "Sample music not found"}, status_code=404)

@app.get("/api/audio_waveform")
def get_audio_waveform(source: str = "auto"):
    """
    Generate downsampled audio envelope data (positive/negative peaks)
    along with parsed detected rhythm note timestamps for interactive waveform scrubbing.
    """
    try:
        custom_p = DATA_DIR / "custom_audio.wav"
        if source == "custom" and custom_p.exists() and custom_p.stat().st_size > 1000:
            target_path = custom_p
        elif source == "sample" or not custom_p.exists() or custom_p.stat().st_size <= 1000:
            target_path = DATA_DIR / "test" / "test_music.ogg"
        else:
            target_path = custom_p

        if not target_path.exists():
            return JSONResponse({"error": "Audio source file not found"}, status_code=404)

        y, sr = librosa.load(str(target_path), sr=22050)
        duration = float(len(y) / sr)
        points = 1000
        step = max(1, len(y) // points)
        pos_peaks = []
        neg_peaks = []
        for i in range(0, len(y), step):
            chunk = y[i:i+step]
            if len(chunk) > 0:
                pos_peaks.append(round(float(np.max(chunk)), 3))
                neg_peaks.append(round(float(np.min(chunk)), 3))

        notes = []
        tja_path = DATA_DIR / "inference" / "inferred_notes.tja"
        if tja_path.exists():
            bpm = 240.0
            measure_duration = 4.0 * (60.0 / bpm)
            current_time = 0.0
            with open(tja_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("BPM:"):
                        try:
                            bpm = float(line.split(":")[1])
                            measure_duration = 4.0 * (60.0 / bpm)
                        except Exception:
                            pass
                    elif line.endswith(","):
                        measure = line[:-1]
                        n = len(measure)
                        if n > 0:
                            char_dur = measure_duration / n
                            for idx, ch in enumerate(measure):
                                if ch in ("1", "2"):
                                    notes.append({
                                        "time": round(current_time + idx * char_dur, 3),
                                        "type": "DON" if ch == "1" else "KA"
                                    })
                            current_time += measure_duration

        return JSONResponse({
            "source": target_path.name,
            "duration": round(duration, 3),
            "sample_rate": sr,
            "points": len(pos_peaks),
            "pos": pos_peaks,
            "neg": neg_peaks,
            "notes": notes
        })
    except Exception as e:
        log_event(f"Error computing audio waveform: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/spectrogram_data")
def get_spectrogram_data(source: str = "auto", n_mels: int = 48):
    """
    Generate normalized Mel-frequency spectrogram matrix data for spectral heatmap visualization.
    """
    try:
        custom_p = DATA_DIR / "custom_audio.wav"
        if source == "custom" and custom_p.exists() and custom_p.stat().st_size > 1000:
            target_path = custom_p
        elif source == "sample" or not custom_p.exists() or custom_p.stat().st_size <= 1000:
            target_path = DATA_DIR / "test" / "test_music.ogg"
        else:
            target_path = custom_p

        if not target_path.exists():
            return JSONResponse({"error": "Audio source file not found"}, status_code=404)

        y, sr = librosa.load(str(target_path), sr=22050)
        duration = float(len(y) / sr)
        # Compute mel spectrogram downsampled for fast transfer and rendering
        # Hop length of ~2205 yields 10 frames per second
        hop_length = max(512, int(sr * 0.1))
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=20,
            fmax=sr // 2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalize to 0.0 - 1.0
        min_db = float(mel_db.min())
        max_db = float(mel_db.max())
        norm_mel = (mel_db - min_db) / (max_db - min_db + 1e-6)
        # Round values to 2 decimals to minimize JSON payload
        norm_mel = np.round(norm_mel, 2).tolist()

        return JSONResponse({
            "source": target_path.name,
            "duration": round(duration, 3),
            "sample_rate": sr,
            "n_mels": n_mels,
            "frames": len(norm_mel[0]) if norm_mel else 0,
            "matrix": norm_mel
        })
    except Exception as e:
        log_event(f"Error computing spectrogram: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)



@app.post("/api/chart/quantize")
async def quantize_charts(
    bpm: float = Form(120.0),
    division: int = Form(16),
    offset: float = Form(0.0)
):
    """
    Quantize inferred note timestamps according to BPM and grid division (4, 8, 12, 16, 24, 32),
    then re-export all supported chart formats (.tja, .osu, .sm, .dat).
    """
    try:
        tja_path = INFERENCE_DIR / "inferred_notes.tja"
        if not tja_path.exists():
            return JSONResponse({"error": "No inferred notes chart found. Please run synthesis first."}, status_code=400)

        # Parse existing notes from .tja
        don_times = []
        ka_times = []
        current_bpm = 240.0
        measure_duration = 4.0 * (60.0 / current_bpm)
        current_time = 0.0

        with open(tja_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("BPM:"):
                    try:
                        current_bpm = float(line.split(":")[1])
                        measure_duration = 4.0 * (60.0 / current_bpm)
                    except Exception:
                        pass
                elif line.endswith(","):
                    measure = line[:-1]
                    n = len(measure)
                    if n > 0:
                        char_dur = measure_duration / n
                        for idx, ch in enumerate(measure):
                            t = round(current_time + idx * char_dur, 4)
                            if ch == "1":
                                don_times.append(t)
                            elif ch == "2":
                                ka_times.append(t)
                        current_time += measure_duration

        don_arr = np.array(don_times, dtype=np.float64)
        ka_arr = np.array(ka_times, dtype=np.float64)

        # Quantize to target grid
        quantized_don = chart_exporters.quantize_timestamps_to_grid(don_arr, bpm=bpm, division=division, offset_sec=offset)
        quantized_ka = chart_exporters.quantize_timestamps_to_grid(ka_arr, bpm=bpm, division=division, offset_sec=offset)

        # Avoid simultaneous don/ka collisions by prioritizing don
        quantized_ka = np.array([k for k in quantized_ka if not any(abs(k - d) < 1e-4 for d in quantized_don)], dtype=np.float64)

        # Re-export to multi-formats
        class MockSong:
            def __init__(self, samplerate=22050):
                self.samplerate = samplerate

        mock_song = MockSong()
        # Convert seconds to 512-hop frame indices
        don_frames = np.rint(quantized_don * mock_song.samplerate / 512).astype(np.int64)
        ka_frames = np.rint(quantized_ka * mock_song.samplerate / 512).astype(np.int64)

        chart_exporters.export_osu(
            str(INFERENCE_DIR / "inferred_notes.osu"),
            mock_song, don_frames, ka_frames,
            title="ODCNN Quantized Beatmap", bpm=bpm
        )
        chart_exporters.export_sm(
            str(INFERENCE_DIR / "inferred_notes.sm"),
            mock_song, don_frames, ka_frames,
            title="ODCNN Quantized Dance Track", bpm=bpm
        )
        chart_exporters.export_beat_saber(
            str(INFERENCE_DIR / "inferred_notes.dat"),
            mock_song, don_frames, ka_frames, bpm=bpm
        )
        chart_exporters.export_midi(
            str(INFERENCE_DIR / "inferred_notes.mid"),
            mock_song, don_frames, ka_frames,
            title="ODCNN Quantized MIDI Track", bpm=bpm
        )

        # Re-write .tja chart with quantized measures
        # Compute measure structure based on bpm and quantized timestamps
        total_duration = max(
            float(quantized_don[-1]) if len(quantized_don) else 0.0,
            float(quantized_ka[-1]) if len(quantized_ka) else 0.0
        ) + 2.0
        new_measure_dur = 4.0 * (60.0 / bpm)
        num_measures = int(np.ceil(total_duration / new_measure_dur))

        tja_lines = [
            "TITLE: ODCNN Quantized Track",
            "SUBTITLE: Quantized Grid",
            f"BPM: {bpm:.2f}",
            "WAVE: xxx.ogg",
            f"OFFSET: {offset:.3f}",
            "#START"
        ]

        don_set = set(np.round(quantized_don, 4))
        ka_set = set(np.round(quantized_ka, 4))

        for m_idx in range(num_measures):
            m_start = m_idx * new_measure_dur + offset
            measure_chars = []
            for slot in range(division):
                t_slot = round(m_start + slot * (new_measure_dur / division), 4)
                if t_slot in don_set:
                    measure_chars.append("1")
                elif t_slot in ka_set:
                    measure_chars.append("2")
                else:
                    measure_chars.append("0")
            tja_lines.append("".join(measure_chars) + ",")

        tja_lines.append("#END")
        with open(tja_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tja_lines) + "\n")

        log_event(f"Chart quantized successfully @ {bpm:.1f} BPM, division 1/{division}, {len(quantized_don)} Don, {len(quantized_ka)} Ka.")
        return JSONResponse({
            "status": "success",
            "bpm": bpm,
            "division": division,
            "offset": offset,
            "don_count": len(quantized_don),
            "ka_count": len(quantized_ka)
        })
    except Exception as e:
        log_event(f"Quantization error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload custom audio track for onset processing."""
    try:
        dest_path = DATA_DIR / "custom_audio.wav"
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        log_event(f"Custom audio track uploaded successfully: {file.filename} -> {dest_path}")
        return JSONResponse({"status": "success", "filename": file.filename, "size": dest_path.stat().st_size})
    except Exception as e:
        log_event(f"Audio upload error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/action/simulate_stream")
async def simulate_audio_stream(
    energy_threshold: float = Form(0.04),
    chunk_size: int = Form(1024),
    audio_source: str = Form("sample")
):
    """Run real-time simulated chunk-by-chunk streaming onset detector."""
    def run_stream_sim():
        with JOB_LOCK:
            CURRENT_JOB["status"] = "running"
            CURRENT_JOB["job_name"] = "Streaming Audio Simulation"
            CURRENT_JOB["message"] = f"Processing chunks ({chunk_size} samples) at threshold {energy_threshold}"
            CURRENT_JOB["progress"] = 0

        try:
            log_event(f"Starting simulated audio stream processing ({audio_source})...")
            if audio_source == "custom" and (DATA_DIR / "custom_audio.wav").exists():
                src = str(DATA_DIR / "custom_audio.wav")
            else:
                src = str(DATA_DIR / "test" / "test_music.ogg")

            streamer = audio_stream.AudioChunkStreamer.from_file(src, chunk_size=chunk_size)
            detector = audio_stream.StreamingOnsetDetector(energy_threshold=energy_threshold)

            total_chunks = 0
            total_onsets = 0
            for c_idx, t_sec, chunk in streamer.stream_chunks():
                new_onsets = detector.push_chunk(chunk)
                total_chunks += 1
                if new_onsets:
                    total_onsets += len(new_onsets)
                    for onset in new_onsets:
                        log_event(f"[Stream Trigger @ {onset['timestamp']:.2f}s] {onset['type'].upper()} (conf: {onset['confidence']}, centroid: {onset['centroid_hz']}Hz)")

            log_event(f"Stream simulation finished: {total_chunks} chunks evaluated, {total_onsets} onsets detected.")
            with JOB_LOCK:
                CURRENT_JOB["status"] = "completed"
                CURRENT_JOB["progress"] = 100
                CURRENT_JOB["message"] = f"Simulated stream finished: {total_onsets} onsets found"
        except Exception as e:
            log_event(f"Streaming error: {e}")
            with JOB_LOCK:
                CURRENT_JOB["status"] = "error"
                CURRENT_JOB["message"] = f"Stream error: {str(e)}"
        finally:
            time.sleep(1.5)
            with JOB_LOCK:
                if CURRENT_JOB["status"] == "completed":
                    CURRENT_JOB["status"] = "idle"
                    CURRENT_JOB["job_name"] = None
                    CURRENT_JOB["message"] = "Ready"

    threading.Thread(target=run_stream_sim, daemon=True).start()
    return JSONResponse({"status": "Simulation started in background"})


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-time low-latency WebSocket stream receiver.
    Accepts raw binary Float32 PCM audio chunks from browser/microphone and returns onset events.
    """
    await websocket.accept()
    detector = audio_stream.StreamingOnsetDetector(energy_threshold=0.035)
    log_event("WebSocket live audio streaming client connected.")
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                break
            # Convert raw bytes to float32 numpy array
            chunk = np.frombuffer(data, dtype=np.float32)
            if len(chunk) > 0:
                onsets = detector.push_chunk(chunk)
                if onsets:
                    for onset in onsets:
                        log_event(f"[Live Mic Onset] {onset['type'].upper()} at {onset['timestamp']:.2f}s ({onset['centroid_hz']}Hz)")
                    await websocket.send_json({"onsets": onsets})
    except WebSocketDisconnect:
        log_event("WebSocket live audio client disconnected.")
    except Exception as e:
        log_event(f"WebSocket error: {e}")


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ODCNN • Musical Onset Detection & Rhythm Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: rgba(22, 27, 34, 0.85);
      --card-border: rgba(48, 54, 61, 0.7);
      --primary: #f85149;
      --primary-hover: #da3633;
      --secondary: #58a6ff;
      --accent-ka: #388bfd;
      --accent-don: #f85149;
      --accent-green: #2ea043;
      --accent-gold: #d29922;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --font-main: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      background-image:
        radial-gradient(at 0% 0%, rgba(248, 81, 73, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(56, 139, 253, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(46, 160, 67, 0.08) 0px, transparent 50%);
      color: var(--text);
      font-family: var(--font-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(13, 17, 23, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      padding: 1rem 2rem;
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      font-size: 1.8rem;
      filter: drop-shadow(0 0 8px rgba(248, 81, 73, 0.6));
    }
    .brand-title {
      font-weight: 800;
      font-size: 1.3rem;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, #f85149, #58a6ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-badge {
      background: rgba(248, 81, 73, 0.15);
      border: 1px solid var(--primary);
      color: var(--primary);
      font-size: 0.7rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 12px;
    }
    .top-actions {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .system-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.8rem;
      font-family: var(--font-mono);
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-green);
      box-shadow: 0 0 8px var(--accent-green);
    }
    .btn-quit {
      background: rgba(248, 81, 73, 0.15);
      color: var(--primary);
      border: 1px solid var(--primary);
      border-radius: 6px;
      padding: 6px 12px;
      font-weight: 600;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-quit:hover {
      background: var(--primary);
      color: #fff;
    }

    main {
      flex: 1;
      max-width: 1400px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }

    .hero-banner {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      backdrop-filter: blur(8px);
    }
    .hero-banner h1 {
      font-size: 1.6rem;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .hero-banner p {
      color: var(--text-muted);
      font-size: 0.95rem;
    }

    /* Grid Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 1.5rem;
    }

    .col-12 { grid-column: span 12; }
    .col-8 { grid-column: span 8; }
    .col-6 { grid-column: span 6; }
    .col-4 { grid-column: span 4; }

    @media (max-width: 992px) {
      .col-8, .col-6, .col-4 { grid-column: span 12; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      backdrop-filter: blur(8px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.2rem;
      padding-bottom: 0.8rem;
      border-bottom: 1px solid var(--card-border);
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .tooltip-icon {
      color: var(--secondary);
      font-size: 0.8rem;
      cursor: help;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border: 1px solid var(--secondary);
      border-radius: 50%;
    }

    .form-group {
      margin-bottom: 1rem;
    }
    .form-group label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .form-control, select {
      width: 100%;
      background: rgba(13, 17, 23, 0.8);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px 12px;
      color: var(--text);
      font-family: var(--font-main);
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .form-control:focus, select:focus {
      border-color: var(--secondary);
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-family: var(--font-main);
      font-weight: 600;
      font-size: 0.9rem;
      padding: 10px 16px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      transition: all 0.2s;
      width: 100%;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--primary), #e53935);
      color: #fff;
    }
    .btn-primary:hover {
      box-shadow: 0 0 16px rgba(248, 81, 73, 0.4);
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: linear-gradient(135deg, var(--secondary), #1f6feb);
      color: #fff;
    }
    .btn-secondary:hover {
      box-shadow: 0 0 16px rgba(88, 166, 255, 0.4);
      transform: translateY(-1px);
    }
    .btn-success {
      background: linear-gradient(135deg, var(--accent-green), #238636);
      color: #fff;
    }
    .btn-success:hover {
      box-shadow: 0 0 16px rgba(46, 160, 67, 0.4);
      transform: translateY(-1px);
    }

    /* Status indicators */
    .status-pill {
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 12px;
      font-family: var(--font-mono);
      font-weight: 600;
    }
    .pill-ready { background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea043; }
    .pill-missing { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #da3633; }

    /* Audio player styling */
    .audio-player-box {
      background: rgba(13, 17, 23, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 1rem;
      margin-top: 1rem;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    audio {
      width: 100%;
      height: 40px;
    }

    /* Terminal logs */
    .log-box {
      background: #000;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 1rem;
      font-family: var(--font-mono);
      font-size: 0.8rem;
      height: 220px;
      overflow-y: auto;
      color: #7ee787;
      display: flex;
      flex-direction: column-reverse;
      white-space: pre-wrap;
    }

    /* Metrics */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 1rem;
    }
    .metric-card {
      background: rgba(13, 17, 23, 0.7);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }
    .metric-val {
      font-size: 1.3rem;
      font-weight: 700;
      font-family: var(--font-mono);
      color: var(--secondary);
    }
    .metric-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span class="brand-logo">🥁</span>
      <div>
        <div class="brand-title">ODCNN AUTOPILOT STUDIO</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">Neural Musical Onset & Taiko Chart Synthesizer</div>
      </div>
      <span class="brand-badge" id="appVersion">v1.0.0</span>
    </div>

    <div class="top-actions">
      <div class="system-badge" title="Live compute engine status">
        <span class="pulse-dot"></span>
        <span id="deviceStatus">Engine: CPU</span>
      </div>
      <button class="btn-quit" id="btnQuit" onclick="quitServer()" title="Terminate and quit the background server">
        🛑 Quit Server
      </button>
    </div>
  </header>

  <main>
    <!-- Top Hero Banner: High Value Summary -->
    <div class="hero-banner">
      <div>
        <h1>Unified Neural Onset Detection & Synthesis Studio</h1>
        <p>End-to-end rhythm game AI: Extract multi-scale spectrograms, train dual CNNs (Don/Ka), and synthesize Taiko charts.</p>
      </div>
      <div id="jobStatusBox" style="text-align: right;">
        <span class="status-pill pill-ready" id="jobStatusBadge">System: Ready</span>
        <div id="jobDetails" style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">No active background jobs</div>
      </div>
    </div>

    <!-- Metrics Bar -->
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-val" id="metricCpu">0%</div>
        <div class="metric-label">CPU Utilization</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" id="metricRam">0%</div>
        <div class="metric-label">Memory Usage</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" id="metricModels">2 / 2</div>
        <div class="metric-label">Trained Models (Don / Ka)</div>
      </div>
    </div>

    <div class="dashboard-grid">

      <!-- High Value Section 1: Peak Detection & Chart Synthesis -->
      <div class="card col-6">
        <div class="card-header">
          <div class="card-title">
            <span>✨ Step 3: Peak Picking & Taiko Chart Synthesis</span>
            <span class="tooltip-icon" title="Transforms neural onset probabilities into discrete drum notes and exports .tja charts + synthesized audio.">?</span>
          </div>
          <span class="status-pill" id="synthesizeStatusBadge">Status</span>
        </div>

        <form id="synthForm" onsubmit="event.preventDefault(); runSynthesis();">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div class="form-group">
              <label>Onset Delta (Threshold)
                <span class="tooltip-icon" title="Threshold offset above local spectral average for peak qualification.">?</span>
              </label>
              <input type="number" step="0.01" class="form-control" id="synthDelta" value="0.05">
            </div>
            <div class="form-group">
              <label>Wait Frames (Refractory)
                <span class="tooltip-icon" title="Minimum frame separation between consecutive detected notes.">?</span>
              </label>
              <input type="number" class="form-control" id="synthWait" value="3">
            </div>
          </div>

          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div class="form-group">
              <label>Pre/Post Max Window</label>
              <input type="number" class="form-control" id="synthPreMax" value="1">
            </div>
            <div class="form-group">
              <label>Pre/Post Avg Window</label>
              <input type="number" class="form-control" id="synthPreAvg" value="4">
            </div>
          </div>

          <button type="submit" class="btn btn-primary" id="btnSynthesize">
            🎵 Synthesize Taiko Audio & Generate .TJA
          </button>
        </form>

        <div class="audio-player-box">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; font-size: 0.85rem;">Synthesized Audio Preview</span>
            <a href="/api/download/audio" download class="btn-quit" style="text-decoration: none; padding: 3px 8px;">Download WAV</a>
          </div>
          <audio controls id="synthesizedAudioPlayer">
            <source src="/api/download/audio" type="audio/wav">
            Your browser does not support the audio tag.
          </audio>
          <div style="margin-top: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.8rem; color: var(--text-muted);">Taiko Sanjiro Chart File:</span>
            <a href="/api/download/tja" download class="btn-quit" style="text-decoration: none; padding: 3px 8px;">Download .TJA</a>
          </div>
          <div style="margin-top: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.8rem; color: var(--text-muted);">osu! Taiko Beatmap (.osu):</span>
            <a href="/api/download/osu" download class="btn-quit" style="text-decoration: none; padding: 3px 8px;">Download .OSU</a>
          </div>
          <div style="margin-top: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.8rem; color: var(--text-muted);">StepMania Dance Chart (.sm):</span>
            <a href="/api/download/sm" download class="btn-quit" style="text-decoration: none; padding: 3px 8px;">Download .SM</a>
          </div>
          <div style="margin-top: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.8rem; color: var(--text-muted);">Beat Saber Track (.dat):</span>
            <a href="/api/download/beatsaber" download class="btn-quit" style="text-decoration: none; padding: 3px 8px;">Download .DAT</a>
          </div>
          <div style="margin-top: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.8rem; color: var(--text-muted);">Standard MIDI Track (.mid):</span>
            <a href="/api/download/midi" download id="downloadMidiBtn" class="btn-quit" style="text-decoration: none; padding: 3px 8px;">Download .MID</a>
          </div>
        </div>
      </div>

      <!-- High Value Section 2: Neural Inference Engine -->
      <div class="card col-6">
        <div class="card-header">
          <div class="card-title">
            <span>⚡ Step 2: CNN Onset Inference</span>
            <span class="tooltip-icon" title="Feeds preprocessed audio features through the trained Don and Ka convolutional neural networks.">?</span>
          </div>
          <span class="status-pill" id="inferStatusBadge">Status</span>
        </div>

        <form id="inferForm" onsubmit="event.preventDefault(); runInference();">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div class="form-group">
              <label>Inference Target Models
                <span class="tooltip-icon" title="Select whether to infer Don (red notes), Ka (blue rimshots), or both.">?</span>
              </label>
              <select id="inferTarget">
                <option value="both">Both Models (Don + Ka)</option>
                <option value="don">Don (Bass/Center Drum)</option>
                <option value="ka">Ka (Rimshot/Edge Drum)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Model Architecture
                <span class="tooltip-icon" title="Choose Standard CNN (Schlüter & Böck) or Modern CRNN with Attention.">?</span>
              </label>
              <select id="inferArch">
                <option value="cnn">Standard CNN (Schlüter & Böck)</option>
                <option value="crnn">Modern CRNN (CNN + BiGRU + Attention)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Execution Engine
                <span class="tooltip-icon" title="Choose PyTorch or portable ONNX Runtime.">?</span>
              </label>
              <select id="inferEngine">
                <option value="pytorch">PyTorch (Torch Script)</option>
                <option value="onnx">ONNX Runtime (High Throughput)</option>
              </select>
            </div>
          </div>

          <div style="display: flex; gap: 8px;">
            <button type="submit" class="btn btn-secondary" id="btnInfer">
              🚀 Run Neural Inference
            </button>
            <button type="button" class="btn-quit" onclick="runExportONNX()" style="white-space: nowrap; padding: 0 12px;" title="Export PyTorch models to ONNX graphs">
              ⚡ Export ONNX
            </button>
          </div>
        </form>

        <div class="audio-player-box" style="margin-top: 1.5rem;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; font-size: 0.85rem;">Original Test Music Track</span>
            <a href="/api/download/sample_music" download class="btn-quit" style="text-decoration: none; padding: 3px 8px;">Download Source</a>
          </div>
          <audio controls>
            <source src="/api/download/sample_music" type="audio/ogg">
          </audio>
        </div>
      </div>

      <!-- Section 3: Audio Preprocessing & Feature Extraction -->
      <div class="card col-6">
        <div class="card-header">
          <div class="card-title">
            <span>🎼 Step 1: Preprocessing & Mel-Spectrogram Extraction</span>
            <span class="tooltip-icon" title="Extracts 3-channel FFT mel-frequency filterbank representations across multiple window sizes.">?</span>
          </div>
          <span class="status-pill" id="prepStatusBadge">Status</span>
        </div>

        <form id="prepForm" onsubmit="event.preventDefault(); runPreprocessing();">
          <div class="form-group">
            <label>Dataset Target
              <span class="tooltip-icon" title="Choose which audio dataset partition to pickle and transform.">?</span>
            </label>
            <select id="prepTarget">
              <option value="test">Test Track (data/test)</option>
              <option value="reduced">Reduced Training Songs (data/train_reduced)</option>
              <option value="val">Validation Track (data/validation)</option>
            </select>
          </div>

          <button type="submit" class="btn btn-success" id="btnPrep">
            ⚙️ Extract Spectrogram Features
          </button>
        </form>
      </div>

      <!-- Section 4: CNN Training Center -->
      <div class="card col-6">
        <div class="card-header">
          <div class="card-title">
            <span>🧠 Model Training Center</span>
            <span class="tooltip-icon" title="Train the Schlüter & Böck CNN architecture on preprocessed song features.">?</span>
          </div>
          <span class="status-pill" id="trainStatusBadge">Status</span>
        </div>

        <form id="trainForm" onsubmit="event.preventDefault(); runTraining();">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div class="form-group">
              <label>Target Model
                <span class="tooltip-icon" title="Select whether to train Don (bass) or Ka (rim) model.">?</span>
              </label>
              <select id="trainModelType">
                <option value="don">Don (Red Drum)</option>
                <option value="ka">Ka (Blue Rim)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Epochs</label>
              <input type="number" class="form-control" id="trainEpochs" value="5" min="1" max="100">
            </div>
          </div>

          <div class="form-group">
            <label>Batch Size</label>
            <input type="number" class="form-control" id="trainBatchSize" value="64" min="8" max="512">
          </div>

          <button type="submit" class="btn btn-secondary" id="btnTrain">
            🔥 Train Convolutional Network
          </button>
        </form>
      </div>


      <!-- Section: BPM Tap Tempo & Beat Grid Snapping -->
      <div class="card col-12" id="bpmQuantizeCard">
        <div class="card-header">
          <div class="card-title">
            <span>⏱️ BPM Tap Tempo & Beat Grid Quantizer</span>
            <span class="tooltip-icon" title="Tap along to the music to detect BPM or set a manual tempo, then snap detected onsets to standard rhythmic subdivisions (quarter, eighth, sixteenth notes).">?</span>
          </div>
          <span class="status-pill pill-ready" id="bpmStatusBadge">Grid Ready</span>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; align-items: start;">
          <div>
            <h4 style="font-size: 0.95rem; margin-bottom: 8px;">1. Tap Tempo Estimator</h4>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 12px;">
              Tap rhythmically on the button below (or press Spacebar when active) to calculate live BPM.
            </p>
            <div style="display: flex; gap: 10px; align-items: center;">
              <button type="button" class="btn btn-secondary" id="btnTapTempo" onclick="handleTapTempo()" style="flex: 1; font-weight: 700; padding: 10px;">
                🥁 TAP TEMPO
              </button>
              <button type="button" class="btn-quit" onclick="resetTapTempo()" title="Reset tap tempo history">
                Reset
              </button>
            </div>
            <div style="margin-top: 10px; display: flex; align-items: baseline; gap: 8px;">
              <span style="font-size: 1.8rem; font-weight: 800; color: var(--warning);" id="displayBpm">120.0</span>
              <span style="font-size: 0.9rem; color: var(--text-muted);">BPM</span>
              <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: auto;" id="tapCountLabel">0 taps</span>
            </div>
          </div>

          <div>
            <h4 style="font-size: 0.95rem; margin-bottom: 8px;">2. Beat Grid Settings</h4>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 12px;">
              Configure tempo, rhythmic division per measure, and start offset.
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
              <div class="form-group" style="margin-bottom: 8px;">
                <label>Target BPM</label>
                <input type="number" step="0.1" class="form-control" id="gridBpmInput" value="120.0">
              </div>
              <div class="form-group" style="margin-bottom: 8px;">
                <label>Subdivision</label>
                <select id="gridDivisionSelect">
                  <option value="4">1/4 Measure (Quarter Notes)</option>
                  <option value="8">1/8 Measure (Eighth Notes)</option>
                  <option value="12">1/12 Measure (Triplets)</option>
                  <option value="16" selected>1/16 Measure (Sixteenths)</option>
                  <option value="24">1/24 Measure (Sextuplets)</option>
                  <option value="32">1/32 Measure (Thirty-seconds)</option>
                </select>
              </div>
            </div>
            <div class="form-group" style="margin-bottom: 0;">
              <label>Offset (Seconds)</label>
              <input type="number" step="0.01" class="form-control" id="gridOffsetInput" value="0.0">
            </div>
          </div>

          <div>
            <h4 style="font-size: 0.95rem; margin-bottom: 8px;">3. Quantize & Re-Export</h4>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 12px;">
              Align raw neural timestamps to the rhythmic grid and update .tja, .osu, .sm, and .dat charts.
            </p>
            <button type="button" class="btn btn-primary" id="btnQuantizeChart" onclick="quantizeAndReexport()" style="width: 100%; margin-bottom: 8px;">
              📐 Quantize Charts to Grid
            </button>
            <div id="quantizeResultBadge" style="font-size: 0.8rem; color: var(--text-muted);">
              Charts aligned: Ready to snap.
            </div>
          </div>
        </div>
      </div>

      <!-- Section 5: Live Streaming & Custom Audio -->
      <div class="card col-12">
        <div class="card-header">
          <div class="card-title">
            <span>🎙️ Live Audio Input & Real-Time Stream Simulator</span>
            <span class="tooltip-icon" title="Stream live audio chunks directly through the sliding buffer onset detector or upload custom music tracks for instantaneous analysis.">?</span>
          </div>
          <span class="status-pill pill-ready">Streaming Engine Ready</span>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
          <div>
            <h4 style="font-size: 0.95rem; margin-bottom: 8px;">Upload Custom Music File</h4>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 12px;">
              Upload any WAV or OGG file to run onset detection and synthesis without modifying the test suite.
            </p>
            <div style="display: flex; gap: 10px;">
              <input type="file" id="customAudioFile" class="form-control" accept=".wav,.ogg,.mp3">
              <button class="btn btn-primary" onclick="uploadAudioTrack()" style="width: auto; white-space: nowrap;">Upload Track</button>
            </div>
            <div id="uploadStatusMsg" style="font-size: 0.75rem; color: var(--accent-cyan); margin-top: 6px;"></div>
          </div>

          <div>
            <h4 style="font-size: 0.95rem; margin-bottom: 8px;">Simulate Real-Time Sliding Buffer</h4>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 12px;">
              Test low-latency audio chunk processing (1024 samples/chunk) with instantaneous RMS and spectral analysis.
            </p>
            <div style="display: flex; gap: 10px;">
              <select id="simAudioSource" class="form-control">
                <option value="sample">Built-in Test Track</option>
                <option value="custom">Custom Uploaded Track</option>
              </select>
              <button class="btn btn-primary" onclick="runStreamSimulation()" style="width: auto; white-space: nowrap;">Simulate Stream</button>
            </div>
          </div>

          <div>
            <h4 style="font-size: 0.95rem; margin-bottom: 8px;">Live Microphone Stream</h4>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 12px;">
              Direct low-latency WebSocket audio stream directly from your browser microphone.
            </p>
            <div style="display: flex; gap: 10px; align-items: center;">
              <button id="btnToggleMic" class="btn btn-primary" onclick="toggleMicrophoneStream()" style="width: auto; white-space: nowrap;">🎙️ Start Live Mic</button>
              <div id="micIndicator" style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.8rem; color: var(--text-muted);">
                <span id="hitBadgeDon" style="display: none; padding: 2px 6px; border-radius: 4px; background: var(--accent-don); color: white; font-weight: bold;">DON</span>
                <span id="hitBadgeKa" style="display: none; padding: 2px 6px; border-radius: 4px; background: var(--accent-ka); color: white; font-weight: bold;">KA</span>
              </div>
            </div>
            <div id="micStatusMsg" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">Microphone idle</div>
          </div>
        </div>
      </div>

      <!-- Section 6: Interactive Waveform Visualizer & Onset Scrubber -->
      <div class="card col-12">
        <div class="card-header">
          <div class="card-title">
            <span>🌊 Interactive Audio Waveform & Rhythm Onset Scrubber</span>
            <span class="tooltip-icon" title="High-precision audio envelope visualization with overlaid neural onset detections (Red Don / Blue Ka). Click anywhere on the waveform or on an onset marker to jump audio playback directly.">?</span>
          </div>
          <div style="display: flex; gap: 8px; align-items: center;">
            <button class="btn btn-secondary" onclick="loadWaveformData()" style="padding: 4px 10px; font-size: 0.8rem;">↻ Reload Waveform</button>
            <span class="status-pill pill-ready" id="waveStatusBadge">Ready</span>
          </div>
        </div>

        <div style="position: relative; width: 100%; margin-top: 10px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.85rem;">
            <div>
              <span style="color: var(--text-muted);">Current Track:</span>
              <strong id="waveTrackName">test_music.ogg</strong>
              <span style="margin-left: 12px; color: var(--text-muted);">Duration:</span>
              <strong id="waveDuration">0:00</strong>
              <span style="margin-left: 12px; color: var(--text-muted);">Detected Notes:</span>
              <strong id="waveNoteCount">0</strong>
              (<span style="color: var(--accent-don); font-weight: bold;">● Don</span> / <span style="color: var(--accent-ka); font-weight: bold;">● Ka</span>)
            </div>
            <div style="color: var(--accent-gold); font-size: 0.8rem;" id="waveScrubberTime">
              Playback: 0.00s / 0.00s
            </div>
          </div>

          <div style="position: relative; background: #0c1017; border: 1px solid var(--border-color); border-radius: 6px; overflow: hidden; height: 140px; cursor: pointer;" id="waveformContainer" onclick="handleWaveformClick(event)">
            <canvas id="waveformCanvas" width="1200" height="140" style="width: 100%; height: 100%; display: block;"></canvas>
            <div id="playheadBar" style="position: absolute; top: 0; left: 0%; width: 2px; height: 100%; background: #ffd700; pointer-events: none; transition: left 0.05s linear; box-shadow: 0 0 8px #ffd700;"></div>
          </div>
          <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">
            💡 Click anywhere inside the waveform canvas to scrub audio playback time. Hover to view note timestamps.
          </p>
        </div>

        <!-- Real-Time Frequency Heatmap Visualizer -->
        <div style="position: relative; width: 100%; margin-top: 1.5rem; border-top: 1px solid var(--card-border); padding-top: 1rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.85rem;">
            <div>
              <span style="font-weight: 600;">🔥 Mel-Frequency Spectral Heatmap</span>
              <span style="color: var(--text-muted); font-size: 0.75rem; margin-left: 8px;">(48 Mel Bins • 20Hz - 11kHz)</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
              <button class="btn-quit" onclick="loadSpectrogramData()" style="padding: 2px 8px; font-size: 0.75rem;">↻ Reload Heatmap</button>
              <span class="status-pill pill-ready" id="spectroStatusBadge" style="font-size: 0.7rem;">Spectrogram Ready</span>
            </div>
          </div>

          <div style="position: relative; background: #000; border: 1px solid var(--border-color); border-radius: 6px; overflow: hidden; height: 120px; cursor: pointer;" id="spectrogramContainer" onclick="handleWaveformClick(event)">
            <canvas id="spectrogramCanvas" width="1200" height="120" style="width: 100%; height: 100%; display: block;"></canvas>
            <div id="spectroPlayheadBar" style="position: absolute; top: 0; left: 0%; width: 2px; height: 100%; background: #ffd700; pointer-events: none; transition: left 0.05s linear; box-shadow: 0 0 8px #ffd700;"></div>
          </div>
          <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">
            Spectral power distribution: Deep blue (low energy) → Violet → Amber → Bright White (transient onsets).
          </p>
        </div>
      </div>

      <!-- Section 7: Real-Time Engine Logs -->
      <div class="card col-12">
        <div class="card-header">
          <div class="card-title">
            <span>📜 Real-Time Process & Diagnostic Logs</span>
            <span class="tooltip-icon" title="Streaming logs from the ODCNN neural network and audio processing pipeline.">?</span>
          </div>
          <button class="btn-quit" onclick="fetchLogs()" style="padding: 4px 10px;">Refresh Logs</button>
        </div>

        <div class="log-box" id="logBox">Loading diagnostic logs...</div>
      </div>

    </div>
  </main>

  <script>
    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        document.getElementById('appVersion').textContent = 'v' + data.version;
        document.getElementById('metricCpu').textContent = data.cpu_usage + '%';
        document.getElementById('metricRam').textContent = data.ram_usage + '%';
        document.getElementById('deviceStatus').textContent = 'Engine: ' + data.device_name;

        // Job details
        const job = data.job;
        const jobBadge = document.getElementById('jobStatusBadge');
        const jobDetails = document.getElementById('jobDetails');

        if (job.status === 'running') {
          jobBadge.className = 'status-pill pill-missing';
          jobBadge.textContent = 'RUNNING: ' + job.job_name;
          jobDetails.textContent = job.message;
        } else {
          jobBadge.className = 'status-pill pill-ready';
          jobBadge.textContent = 'System: Ready';
          jobDetails.textContent = job.message;
        }

        // Artifact badges
        const arts = data.artifacts;
        updateBadge('prepStatusBadge', arts.test_pickle, 'Features Cached', 'Pickle Missing');
        updateBadge('trainStatusBadge', arts.don_model && arts.ka_model, 'Models Ready', 'Models Incomplete');
        updateBadge('inferStatusBadge', arts.don_inference && arts.ka_inference, 'Inference Complete', 'Needs Inference');
        updateBadge('synthesizeStatusBadge', arts.synthesized_wav && arts.synthesized_tja, 'Chart Ready', 'Unsynthesized');

      } catch (err) {
        console.error('Status fetch error:', err);
      }
    }

    function updateBadge(id, ok, okText, failText) {
      const el = document.getElementById(id);
      if (!el) return;
      if (ok) {
        el.className = 'status-pill pill-ready';
        el.textContent = okText;
      } else {
        el.className = 'status-pill pill-missing';
        el.textContent = failText;
      }
    }

    async function fetchLogs() {
      try {
        const res = await fetch('/api/logs');
        const data = await res.json();
        const box = document.getElementById('logBox');
        box.textContent = data.logs.slice().reverse().join('\\n');
      } catch (err) {
        console.error('Logs fetch error:', err);
      }
    }

    async function runPreprocessing() {
      const target = document.getElementById('prepTarget').value;
      const formData = new FormData();
      formData.append('dataset', target);
      await fetch('/api/action/preprocess', { method: 'POST', body: formData });
      setTimeout(updateStatus, 500);
    }

    async function runTraining() {
      const modelType = document.getElementById('trainModelType').value;
      const epochs = document.getElementById('trainEpochs').value;
      const batchSize = document.getElementById('trainBatchSize').value;

      const formData = new FormData();
      formData.append('model_type', modelType);
      formData.append('epochs', epochs);
      formData.append('batch_size', batchSize);
      await fetch('/api/action/train', { method: 'POST', body: formData });
      setTimeout(updateStatus, 500);
    }

    async function runInference() {
      const target = document.getElementById('inferTarget').value;
      const engine = document.getElementById('inferEngine').value;
      const arch = document.getElementById('inferArch').value;
      const formData = new FormData();
      formData.append('model_type', target);
      formData.append('engine', engine);
      formData.append('arch', arch);
      await fetch('/api/action/infer', { method: 'POST', body: formData });
      setTimeout(updateStatus, 500);
    }

    async function runExportONNX() {
      const target = document.getElementById('inferTarget').value;
      const formData = new FormData();
      formData.append('target', target);
      await fetch('/api/action/export_onnx', { method: 'POST', body: formData });
      setTimeout(updateStatus, 500);
    }

    async function runSynthesis() {
      const formData = new FormData();
      formData.append('delta', document.getElementById('synthDelta').value);
      formData.append('wait', document.getElementById('synthWait').value);
      formData.append('pre_max', document.getElementById('synthPreMax').value);
      formData.append('pre_avg', document.getElementById('synthPreAvg').value);
      await fetch('/api/action/synthesize', { method: 'POST', body: formData });
      setTimeout(() => {
        updateStatus();
        document.getElementById('synthesizedAudioPlayer').load();
      }, 1000);
    }

    async function quitServer() {
      if (confirm('Are you sure you want to stop the ODCNN background server?')) {
        await fetch('/api/action/quit', { method: 'POST' });
        alert('Server is shutting down.');
      }
    }

    // Polling intervals
    setInterval(updateStatus, 2000);
    setInterval(fetchLogs, 2500);
    updateStatus();
    fetchLogs();

    async function uploadAudioTrack() {
      const fileInput = document.getElementById('customAudioFile');
      if (!fileInput.files.length) {
        alert('Please select an audio file to upload.');
        return;
      }
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      const msg = document.getElementById('uploadStatusMsg');
      msg.textContent = 'Uploading...';
      try {
        const res = await fetch('/api/upload_audio', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (res.ok) {
          msg.textContent = 'Upload successful: ' + data.filename;
          msg.style.color = 'var(--accent-green)';
        } else {
          msg.textContent = 'Upload failed: ' + (data.error || 'Unknown error');
          msg.style.color = 'var(--accent-don)';
        }
      } catch (e) {
        msg.textContent = 'Upload error: ' + e.message;
        msg.style.color = 'var(--accent-don)';
      }
    }

    async function runStreamSimulation() {
      const source = document.getElementById('simAudioSource').value;
      const formData = new FormData();
      formData.append('audio_source', source);
      formData.append('energy_threshold', 0.04);
      formData.append('chunk_size', 1024);

      try {
        const res = await fetch('/api/action/simulate_stream', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        alert(data.status || 'Simulation initiated');
        fetchLogs();
      } catch (e) {
        alert('Stream simulation failed: ' + e.message);
      }
    }

    let micWs = null;
    let audioContext = null;
    let micStream = null;
    let micProcessor = null;

    async function toggleMicrophoneStream() {
      const btn = document.getElementById('btnToggleMic');
      const statusMsg = document.getElementById('micStatusMsg');

      if (micWs) {
        // Stop streaming
        if (micProcessor) {
          micProcessor.disconnect();
          micProcessor = null;
        }
        if (micStream) {
          micStream.getTracks().forEach(track => track.stop());
          micStream = null;
        }
        if (audioContext) {
          audioContext.close();
          audioContext = null;
        }
        micWs.close();
        micWs = null;
        btn.textContent = '🎙️ Start Live Mic';
        btn.className = 'btn btn-primary';
        statusMsg.textContent = 'Microphone idle';
        statusMsg.style.color = 'var(--text-muted)';
        return;
      }

      // Start streaming
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        micWs = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);

        micWs.onopen = async () => {
          statusMsg.textContent = 'Connecting microphone...';
          audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
          micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const source = audioContext.createMediaStreamSource(micStream);
          micProcessor = audioContext.createScriptProcessor(1024, 1, 1);

          micProcessor.onaudioprocess = (e) => {
            if (micWs && micWs.readyState === WebSocket.OPEN) {
              const inputData = e.inputBuffer.getChannelData(0);
              micWs.send(inputData.buffer);
            }
          };

          source.connect(micProcessor);
          micProcessor.connect(audioContext.destination);

          btn.textContent = '⏹️ Stop Live Mic';
          btn.className = 'btn btn-danger';
          statusMsg.textContent = 'Live stream active (44.1kHz)';
          statusMsg.style.color = 'var(--accent-green)';
        };

        micWs.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          if (msg.onsets && msg.onsets.length > 0) {
            msg.onsets.forEach(onset => {
              triggerVisualHit(onset.type);
            });
          }
        };

        micWs.onclose = () => {
          statusMsg.textContent = 'WebSocket disconnected';
        };

      } catch (err) {
        statusMsg.textContent = 'Microphone error: ' + err.message;
        statusMsg.style.color = 'var(--accent-don)';
        micWs = null;
      }
    }

    function triggerVisualHit(type) {
      const badgeDon = document.getElementById('hitBadgeDon');
      const badgeKa = document.getElementById('hitBadgeKa');
      if (type === 'don') {
        badgeDon.style.display = 'inline-block';
        setTimeout(() => { badgeDon.style.display = 'none'; }, 200);
      } else {
        badgeKa.style.display = 'inline-block';
        setTimeout(() => { badgeKa.style.display = 'none'; }, 200);
      }
    }

    // --- Interactive Waveform & Spectrogram Visualizers ---
    let waveformCache = null;
    let spectrogramCache = null;

    async function loadWaveformData() {
      const badge = document.getElementById('waveStatusBadge');
      badge.textContent = 'Loading...';
      try {
        const res = await fetch('/api/audio_waveform');
        if (!res.ok) throw new Error('Failed to load waveform data');
        waveformCache = await res.json();

        document.getElementById('waveTrackName').textContent = waveformCache.source;
        document.getElementById('waveDuration').textContent = formatSeconds(waveformCache.duration);
        document.getElementById('waveNoteCount').textContent = waveformCache.notes ? waveformCache.notes.length : 0;
        badge.className = 'status-pill pill-ready';
        badge.textContent = 'Waveform Ready';

        renderWaveformCanvas();
      } catch (err) {
        console.error('Waveform error:', err);
        badge.className = 'status-pill pill-missing';
        badge.textContent = 'Error Loading';
      }
    }

    async function loadSpectrogramData() {
      const badge = document.getElementById('spectroStatusBadge');
      badge.textContent = 'Loading...';
      try {
        const res = await fetch('/api/spectrogram_data');
        if (!res.ok) throw new Error('Failed to load spectrogram data');
        spectrogramCache = await res.json();
        badge.className = 'status-pill pill-ready';
        badge.textContent = 'Spectrogram Ready';
        renderSpectrogramCanvas();
      } catch (err) {
        console.error('Spectrogram error:', err);
        badge.className = 'status-pill pill-missing';
        badge.textContent = 'Error Loading';
      }
    }

    function renderSpectrogramCanvas() {
      if (!spectrogramCache || !spectrogramCache.matrix) return;
      const canvas = document.getElementById('spectrogramCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;

      const matrix = spectrogramCache.matrix; // [n_mels, frames]
      const nMels = matrix.length;
      const frames = matrix[0] ? matrix[0].length : 0;
      if (nMels === 0 || frames === 0) return;

      const cellW = width / frames;
      const cellH = height / nMels;

      for (let m = 0; m < nMels; m++) {
        // Mel bins: lowest freq at bottom, highest at top
        const y = height - (m + 1) * cellH;
        for (let f = 0; f < frames; f++) {
          const val = matrix[m][f]; // 0.0 to 1.0
          const x = f * cellW;

          // Magma/Inferno heat map: dark purple -> magenta -> orange -> yellow -> white
          let r = 0, g = 0, b = 0;
          if (val < 0.25) {
            r = Math.floor(val * 4 * 80);
            g = 0;
            b = Math.floor(val * 4 * 160 + 20);
          } else if (val < 0.5) {
            const v = (val - 0.25) * 4;
            r = Math.floor(80 + v * 120);
            g = Math.floor(v * 40);
            b = Math.floor(180 - v * 80);
          } else if (val < 0.75) {
            const v = (val - 0.5) * 4;
            r = Math.floor(200 + v * 55);
            g = Math.floor(40 + v * 150);
            b = 20;
          } else {
            const v = (val - 0.75) * 4;
            r = 255;
            g = Math.floor(190 + v * 65);
            b = Math.floor(20 + v * 235);
          }

          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(x, y, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    }

    function formatSeconds(secs) {
      const m = Math.floor(secs / 60);
      const s = Math.floor(secs % 60);
      return `${m}:${s < 10 ? '0' : ''}${s}`;
    }

    function renderWaveformCanvas() {
      if (!waveformCache) return;
      const canvas = document.getElementById('waveformCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      const midY = height / 2;

      ctx.clearRect(0, 0, width, height);

      // Background subtle grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(width, midY);
      ctx.stroke();

      const pos = waveformCache.pos || [];
      const neg = waveformCache.neg || [];
      const numPoints = pos.length;
      if (numPoints === 0) return;

      const stepX = width / numPoints;

      // Draw Waveform Envelope
      ctx.fillStyle = '#00d2ff';
      for (let i = 0; i < numPoints; i++) {
        const x = i * stepX;
        const topH = Math.max(1, pos[i] * (midY - 4));
        const botH = Math.max(1, Math.abs(neg[i]) * (midY - 4));
        ctx.fillRect(x, midY - topH, Math.max(1, stepX - 0.5), topH + botH);
      }

      // Draw Onset Markers
      const notes = waveformCache.notes || [];
      const duration = waveformCache.duration || 1.0;

      notes.forEach(note => {
        const nx = (note.time / duration) * width;
        ctx.fillStyle = note.type === 'DON' ? '#ff3b30' : '#007aff';
        ctx.fillRect(nx - 1, 0, 2, height);

        // Small top dot
        ctx.beginPath();
        ctx.arc(nx, 8, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function handleWaveformClick(event) {
      if (!waveformCache || !waveformCache.duration) return;
      const container = event.currentTarget || document.getElementById('waveformContainer');
      const rect = container.getBoundingClientRect();
      const clickX = event.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, clickX / rect.width));
      const targetTime = ratio * waveformCache.duration;

      // Update playhead positions visually
      const pct = (ratio * 100) + '%';
      const playhead = document.getElementById('playheadBar');
      if (playhead) playhead.style.left = pct;
      const spectroPlayhead = document.getElementById('spectroPlayheadBar');
      if (spectroPlayhead) spectroPlayhead.style.left = pct;

      // Update time indicator
      document.getElementById('waveScrubberTime').textContent =
        `Playback: ${targetTime.toFixed(2)}s / ${waveformCache.duration.toFixed(2)}s`;

      // Update HTML5 audio player if active
      const player = document.getElementById('synthesizedAudioPlayer');
      if (player && !isNaN(player.duration) && player.duration > 0) {
        player.currentTime = Math.min(targetTime, player.duration);
      }
    }

    // Sync playhead with audio player playback
    document.addEventListener('DOMContentLoaded', () => {
      loadWaveformData();
      loadSpectrogramData();
      const player = document.getElementById('synthesizedAudioPlayer');
      if (player) {
        player.addEventListener('timeupdate', () => {
          if (!waveformCache || !waveformCache.duration) return;
          const pct = `${Math.min(100, Math.max(0, (player.currentTime / waveformCache.duration) * 100))}%`;
          const playhead = document.getElementById('playheadBar');
          if (playhead) playhead.style.left = pct;
          const spectroPlayhead = document.getElementById('spectroPlayheadBar');
          if (spectroPlayhead) spectroPlayhead.style.left = pct;
          const timeText = document.getElementById('waveScrubberTime');
          if (timeText) {
            timeText.textContent = `Playback: ${player.currentTime.toFixed(2)}s / ${waveformCache.duration.toFixed(2)}s`;
          }
        });
      }
    });


    // --- BPM Tap Tempo & Beat Grid Snapping Logic ---
    let tapTimes = [];
    function handleTapTempo() {
      const now = performance.now() / 1000;
      if (tapTimes.length > 0 && (now - tapTimes[tapTimes.length - 1] > 3.0)) {
        tapTimes = [];
      }
      tapTimes.push(now);
      if (tapTimes.length > 8) tapTimes.shift();

      document.getElementById('tapCountLabel').innerText = `${tapTimes.length} tap${tapTimes.length > 1 ? 's' : ''}`;

      if (tapTimes.length >= 2) {
        let diffs = [];
        for (let i = 1; i < tapTimes.length; i++) {
          diffs.push(tapTimes[i] - tapTimes[i - 1]);
        }
        const avgInterval = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        const calculatedBpm = (60.0 / avgInterval);
        const roundedBpm = Math.round(calculatedBpm * 10) / 10;
        document.getElementById('displayBpm').innerText = roundedBpm.toFixed(1);
        document.getElementById('gridBpmInput').value = roundedBpm.toFixed(1);
      }
    }

    function resetTapTempo() {
      tapTimes = [];
      document.getElementById('tapCountLabel').innerText = '0 taps';
      document.getElementById('displayBpm').innerText = '120.0';
      document.getElementById('gridBpmInput').value = '120.0';
    }

    async function quantizeAndReexport() {
      const btn = document.getElementById('btnQuantizeChart');
      const badge = document.getElementById('quantizeResultBadge');
      const bpm = parseFloat(document.getElementById('gridBpmInput').value) || 120.0;
      const division = parseInt(document.getElementById('gridDivisionSelect').value) || 16;
      const offset = parseFloat(document.getElementById('gridOffsetInput').value) || 0.0;

      btn.disabled = true;
      btn.innerText = 'Snapping...';
      badge.innerText = 'Quantizing chart onsets...';

      try {
        const formData = new FormData();
        formData.append('bpm', bpm);
        formData.append('division', division);
        formData.append('offset', offset);

        const res = await fetch('/api/chart/quantize', {
          method: 'POST',
          body: formData
        });

        const data = await res.json();
        if (res.ok) {
          badge.innerHTML = `<span style="color: var(--success); font-weight: 600;">✓ Quantized: ${data.don_count} Don, ${data.ka_count} Ka @ ${data.bpm} BPM</span>`;
          // Reload waveform and charts
          if (typeof loadWaveform === 'function') {
            loadWaveform();
          }
        } else {
          badge.innerHTML = `<span style="color: var(--primary);">⚠️ ${data.error || 'Failed'}</span>`;
        }
      } catch (err) {
        badge.innerHTML = `<span style="color: var(--primary);">⚠️ Network error</span>`;
      } finally {
        btn.disabled = false;
        btn.innerText = '📐 Quantize Charts to Grid';
      }
    }

  </script>
</body>
</html>
"""

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ODCNN Unified Autopilot Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    log_event(f"Starting ODCNN Autopilot Dashboard on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
