"""
Raptur Desktop Launcher
This script launches the Streamlit app in a local browser.
"""
import os
import sys
import subprocess
import webbrowser
import time
import socket

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def get_app_path():
    """Get the path to app.py, handling both development and bundled modes."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, 'app.py')

def main():
    port = find_free_port()
    app_path = get_app_path()
    
    print(f"Starting Raptur on port {port}...")
    print(f"App path: {app_path}")
    
    env = os.environ.copy()
    env['STREAMLIT_SERVER_PORT'] = str(port)
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    process = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', app_path,
         '--server.port', str(port),
         '--server.headless', 'true',
         '--browser.gatherUsageStats', 'false'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(3)
    
    url = f"http://localhost:{port}"
    print(f"Opening browser at {url}")
    webbrowser.open(url)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down Raptur...")
        process.terminate()

if __name__ == "__main__":
    main()
