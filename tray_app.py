#!/usr/bin/env python3
"""
Desktop System Tray Launcher for ODCNN Autopilot Dashboard.
Allows starting, stopping, monitoring, opening the web dashboard in browser,
and quitting the servers cleanly.
"""

import sys
import os
import time
import signal
import webbrowser
import subprocess
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class TrayController:
    def __init__(self, port=8000):
        self.port = port
        self.server_process = None
        self.running = True

    def start_server(self):
        if self.server_process is None or self.server_process.poll() is not None:
            cmd = [sys.executable, str(BASE_DIR / "dashboard.py"), "--port", str(self.port)]
            self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[TrayController] Started dashboard server (PID: {self.server_process.pid}) on port {self.port}")
            time.sleep(1.5)
        else:
            print("[TrayController] Server is already running.")

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            print(f"[TrayController] Stopping dashboard server (PID: {self.server_process.pid})...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            print("[TrayController] Server stopped.")
        else:
            print("[TrayController] No server currently running.")

    def open_browser(self):
        url = f"http://localhost:{self.port}"
        print(f"[TrayController] Opening dashboard in browser: {url}")
        webbrowser.open(url)

    def status(self):
        if self.server_process and self.server_process.poll() is None:
            return f"Running (PID: {self.server_process.pid})"
        return "Stopped"

    def run_cli_tray(self):
        """CLI fallback tray menu for headless/terminal environments."""
        print("=" * 60)
        print("  ODCNN SYSTEM TRAY CONTROLLER")
        print("=" * 60)
        print("1. Start Server")
        print("2. Stop Server")
        print("3. Open Dashboard in Browser")
        print("4. Status")
        print("5. Quit Tray & Shutdown Server")
        print("=" * 60)

        # Autostart server on launch
        self.start_server()

        while self.running:
            try:
                choice = input("\n[ODCNN Tray] Select option (1-5): ").strip()
                if choice == "1":
                    self.start_server()
                elif choice == "2":
                    self.stop_server()
                elif choice == "3":
                    self.open_browser()
                elif choice == "4":
                    print(f"Server Status: {self.status()}")
                elif choice == "5" or choice.lower() in ("q", "quit", "exit"):
                    print("Exiting tray controller...")
                    self.stop_server()
                    self.running = False
                    break
            except (KeyboardInterrupt, EOFError):
                print("\nReceived termination signal.")
                self.stop_server()
                break

if __name__ == "__main__":
    controller = TrayController()
    controller.run_cli_tray()
