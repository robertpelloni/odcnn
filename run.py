#!/usr/bin/env python3
"""
Unified CLI Entry Point for ODCNN Autopilot Studio.
Provides unified commands for training, neural inference, chart synthesis,
ONNX model export, real-time stream simulation, and web studio launching.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="odcnn",
        description="ODCNN Autopilot: Convolutional Neural Musical Onset Detection & Taiko Chart Synthesizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # studio subcommand
    studio_parser = subparsers.add_parser("studio", help="Launch the unified web dashboard studio")
    studio_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    studio_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    studio_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Train Don or Ka CNN model")
    train_parser.add_argument("--target", choices=["don", "ka"], required=True, help="Target drum note model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")

    # infer subcommand
    infer_parser = subparsers.add_parser("infer", help="Run neural inference on preprocessed audio features")
    infer_parser.add_argument("--target", choices=["don", "ka", "both"], default="both", help="Target model inference")
    infer_parser.add_argument("--engine", choices=["torch", "onnx"], default="torch", help="Execution engine")

    # synthesize subcommand
    synth_parser = subparsers.add_parser("synthesize", help="Perform peak picking and generate Taiko audio + charts")
    synth_parser.add_argument("--delta", type=float, default=0.05, help="Peak picking threshold delta")
    synth_parser.add_argument("--wait", type=int, default=3, help="Refractory frame wait")
    synth_parser.add_argument("--pre-max", type=int, default=1, help="Pre-maximum window")
    synth_parser.add_argument("--post-max", type=int, default=1, help="Post-maximum window")
    synth_parser.add_argument("--pre-avg", type=int, default=4, help="Pre-average window")
    synth_parser.add_argument("--post-avg", type=int, default=4, help="Post-average window")

    # export-onnx subcommand
    export_parser = subparsers.add_parser("export-onnx", help="Export trained PyTorch checkpoints to ONNX graphs")
    export_parser.add_argument("--models-dir", default="models", help="Directory containing .pth models")

    # stream-sim subcommand
    stream_parser = subparsers.add_parser("stream-sim", help="Simulate real-time sliding buffer audio streaming")
    stream_parser.add_argument("--file", default="data/test/test_music.ogg", help="Audio file path")
    stream_parser.add_argument("--chunk-size", type=int, default=1024, help="Samples per chunk buffer")
    stream_parser.add_argument("--threshold", type=float, default=0.04, help="RMS energy threshold ratio")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    if args.subcommand == "studio":
        import uvicorn
        from dashboard import app
        print(f"Launching ODCNN Studio on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.subcommand == "train":
        import subprocess
        print(f"Executing training for {args.target} ({args.epochs} epochs)...")
        # Invoke train.py directly
        subprocess.run([sys.executable, "train.py", args.target], check=True)

    elif args.subcommand == "infer":
        import subprocess
        targets = ["don", "ka"] if args.target == "both" else [args.target]
        for t in targets:
            print(f"Running inference for {t} using engine {args.engine}...")
            subprocess.run([sys.executable, "infer.py", t], check=True)

    elif args.subcommand == "synthesize":
        import subprocess
        print(f"Synthesizing charts (delta={args.delta}, wait={args.wait})...")
        subprocess.run([
            sys.executable, "synthesize.py",
            str(args.delta), str(args.wait),
            str(args.pre_max), str(args.post_max),
            str(args.pre_avg), str(args.post_avg)
        ], check=True)

    elif args.subcommand == "export-onnx":
        import export_onnx
        models_dir = Path(args.models_dir)
        don_pth = models_dir / "don_model.pth"
        ka_pth = models_dir / "ka_model.pth"
        if don_pth.exists():
            export_onnx.export_to_onnx(don_pth, models_dir / "don_model.onnx")
            print("Successfully exported don_model.onnx")
        if ka_pth.exists():
            export_onnx.export_to_onnx(ka_pth, models_dir / "ka_model.onnx")
            print("Successfully exported ka_model.onnx")

    elif args.subcommand == "stream-sim":
        import audio_stream
        print(f"Starting stream simulation for {args.file} (chunk size: {args.chunk_size})...")
        streamer = audio_stream.AudioChunkStreamer.from_file(args.file, chunk_size=args.chunk_size)
        detector = audio_stream.StreamingOnsetDetector(energy_threshold=args.threshold)
        total = 0
        for idx, sec, chunk in streamer.stream_chunks():
            onsets = detector.push_chunk(chunk)
            if onsets:
                for o in onsets:
                    total += 1
                    print(f"[{o['timestamp']:.2f}s] Detected {o['type'].upper()} (confidence: {o['confidence']:.2f}, centroid: {o['centroid_hz']}Hz)")
        print(f"Simulation completed: {total} onsets detected.")

if __name__ == "__main__":
    main()
