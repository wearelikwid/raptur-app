"""
Raptur Desktop Launcher
Uses pywebview to create a native window containing the Streamlit UI.
Includes single-instance protection and clean process lifecycle management.
"""
import os
import sys
import time
import socket
import fcntl
import tempfile
import threading
import subprocess
import http.client
import multiprocessing
from pathlib import Path

import webview

LOCK_PATH = Path(tempfile.gettempdir()) / "raptur_launcher.lock"
STREAMLIT_BOOT_TIMEOUT = 60


def acquire_lock():
    """Prevent multiple instances of the app from running simultaneously."""
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
    except BlockingIOError:
        raise SystemExit("Raptur is already running. Close the existing window first.")
    return lock_file


def find_free_port():
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_app_path():
    """Get the path to app.py, handling both development and bundled modes."""
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, "app.py")


def start_streamlit(port):
    """Start the Streamlit server as a subprocess."""
    env = os.environ.copy()
    env.update({
        "STREAMLIT_SERVER_PORT": str(port),
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    })
    return subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        get_app_path(),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_server(port):
    """Wait for the Streamlit server to be ready before showing the window."""
    deadline = time.time() + STREAMLIT_BOOT_TIMEOUT
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/")
            if conn.getresponse().status < 500:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def launch_window(url, process):
    """Create and launch the native window with the Streamlit UI."""
    window_destroyed = threading.Event()
    
    def on_closed():
        """Clean up when the window is closed."""
        window_destroyed.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def monitor_process():
        """Watch the Streamlit process and close window if it dies."""
        process.wait()
        if not window_destroyed.is_set():
            for window in webview.windows:
                window.destroy()

    threading.Thread(target=monitor_process, daemon=True).start()
    
    window = webview.create_window(
        "Raptur",
        url,
        width=1200,
        height=800,
        resizable=True,
        min_size=(800, 600)
    )
    
    webview.start(debug=False)
    on_closed()


def main():
    """Main entry point for the desktop app."""
    multiprocessing.freeze_support()
    
    lock_file = acquire_lock()
    port = find_free_port()
    process = None
    
    try:
        print(f"Starting Raptur on port {port}...")
        process = start_streamlit(port)
        
        print("Waiting for server to be ready...")
        if not wait_for_server(port):
            raise SystemExit("Streamlit server failed to start within timeout.")
        
        print("Launching window...")
        launch_window(f"http://127.0.0.1:{port}", process)
        
    except KeyboardInterrupt:
        print("\nShutting down Raptur...")
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        lock_file.close()
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
