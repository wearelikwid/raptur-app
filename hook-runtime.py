"""
Runtime hook for Raptur macOS app.
Sets up environment variables so bundled ffmpeg/rubberband can find their dylibs.
"""
import os
import sys

def _setup_dyld_paths():
    """Configure DYLD paths for bundled native libraries."""
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
        macos_dir = os.path.dirname(sys.executable)
        frameworks_dir = os.path.join(os.path.dirname(macos_dir), 'Frameworks')
        
        lib_paths = [
            bundle_dir,
            macos_dir,
            frameworks_dir,
            os.path.join(bundle_dir, 'lib'),
        ]
        
        existing_paths = [p for p in lib_paths if os.path.exists(p)]
        
        current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        new_dyld = ':'.join(existing_paths)
        if current_dyld:
            new_dyld = f"{new_dyld}:{current_dyld}"
        os.environ['DYLD_LIBRARY_PATH'] = new_dyld
        
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = new_dyld
        
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{bundle_dir}:{macos_dir}:{current_path}"


_setup_dyld_paths()
