"""
Runtime hook for Raptur macOS app.
Sets up environment variables so bundled ffmpeg/rubberband can find their dylibs.
"""
import os
import sys

def _setup_dyld_paths():
    """Configure DYLD paths for bundled native libraries."""
    if getattr(sys, 'frozen', False):
        # Running as frozen app
        bundle_dir = sys._MEIPASS
        macos_dir = os.path.dirname(sys.executable)
        frameworks_dir = os.path.join(os.path.dirname(macos_dir), 'Frameworks')
        
        # Build library search paths
        lib_paths = [
            bundle_dir,
            macos_dir,
            frameworks_dir,
            os.path.join(bundle_dir, 'lib'),
        ]
        
        existing_paths = [p for p in lib_paths if os.path.exists(p)]
        
        # Set DYLD_LIBRARY_PATH for dynamic library resolution
        current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        new_dyld = ':'.join(existing_paths)
        if current_dyld:
            new_dyld = f"{new_dyld}:{current_dyld}"
        os.environ['DYLD_LIBRARY_PATH'] = new_dyld
        
        # Also set DYLD_FALLBACK_LIBRARY_PATH as backup
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = new_dyld
        
        # Add bundle dir to PATH for ffmpeg/rubberband executables
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{bundle_dir}:{macos_dir}:{current_path}"

_setup_dyld_paths()
