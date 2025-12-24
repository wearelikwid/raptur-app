"""
Raptur Desktop Launcher
Uses pywebview to create a native window containing the Streamlit UI.
Runs Streamlit via streamlit.web.bootstrap.run() in a multiprocessing.Process
to work correctly inside a frozen PyInstaller bundle.
"""
import os
import sys
import time
import socket
import tempfile
import threading
import http.client
import multiprocessing
from pathlib import Path

# macOS requires 'spawn' method for multiprocessing in frozen apps
if sys.platform == "darwin":
    multiprocessing.set_start_method("spawn", force=True)

import webview

LOCK_PATH = Path(tempfile.gettempdir()) / "raptur_launcher.lock"
STREAMLIT_BOOT_TIMEOUT = 90


def acquire_lock():
    """Prevent multiple instances using a simple PID-based lock file."""
    try:
        if LOCK_PATH.exists():
            old_pid = LOCK_PATH.read_text().strip()
            try:
                os.kill(int(old_pid), 0)
                raise SystemExit("Raptur is already running. Close the existing window first.")
            except (ProcessLookupError, ValueError):
                pass
        LOCK_PATH.write_text(str(os.getpid()))
    except Exception as e:
        if "already running" in str(e):
            raise
        pass


def release_lock():
    """Remove the lock file on shutdown."""
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


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


def run_streamlit_server(app_path, port):
    """
    Target function for multiprocessing.Process.
    Runs Streamlit using bootstrap.run() directly - works in frozen PyInstaller apps.
    """
    import streamlit.web.bootstrap as bootstrap
    
    flag_options = {
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "server.fileWatcherType": "none",
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
    }
    
    bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    
    bootstrap.run(
        app_path,
        False,
        [],
        flag_options
    )


def wait_for_server(port):
    """Wait for the Streamlit server to be ready before showing the window."""
    deadline = time.time() + STREAMLIT_BOOT_TIMEOUT
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/")
            resp = conn.getresponse()
            if resp.status < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def launch_window(url, process):
    """Create and launch the native window with the Streamlit UI."""
    window_destroyed = threading.Event()
    
    def cleanup_process():
        """Terminate the Streamlit process."""
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

    def on_closed():
        """Clean up when the window is closed."""
        window_destroyed.set()
        cleanup_process()

    def monitor_process():
        """Watch the Streamlit process and close window if it dies."""
        process.join()
        if not window_destroyed.is_set():
            for window in webview.windows:
                try:
                    window.destroy()
                except Exception:
                    pass

    threading.Thread(target=monitor_process, daemon=True).start()
    
    webview.create_window(
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
    
    acquire_lock()
    port = find_free_port()
    process = None
    
    try:
        print(f"Starting Raptur on port {port}...")
        
        app_path = get_app_path()
        print(f"App path: {app_path}")
        
        process = multiprocessing.Process(
            target=run_streamlit_server,
            args=(app_path, port),
            daemon=True
        )
        process.start()
        
        print("Waiting for server to be ready...")
        if not wait_for_server(port):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            raise SystemExit("Streamlit server failed to start within timeout.")
        
        print("Launching window...")
        launch_window(f"http://127.0.0.1:{port}", process)
        
    except KeyboardInterrupt:
        print("\nShutting down Raptur...")
    finally:
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        release_lock()


if __name__ == "__main__":
    main()
