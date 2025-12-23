# -*- mode: python ; coding: utf-8 -*-
"""
Raptur PyInstaller Spec File - Hardened macOS Bundle Configuration

Handles:
- Native binaries (ffmpeg, rubberband) from staged directories
- X11 library filtering (prevents Pillow crashes)
- Audio processing dependencies
- pywebview/pyobjc for native macOS window

IMPORTANT: Analysis() binaries use 2-tuple format: (source_path, dest_folder)
"""

from PyInstaller.utils.hooks import collect_all, copy_metadata, collect_data_files, collect_dynamic_libs
import os
import glob

block_cipher = None

# =============================================================================
# CONFIGURATION
# =============================================================================

# X11 libraries to exclude (cause crashes on macOS)
X11_EXCLUDE_PATTERNS = [
    'libxcb', 'libxau', 'libxdmcp', 'libx11', 'libxext', 'libxrender',
    'libxfixes', 'libxcursor', 'libxi', 'libxrandr', 'libxinerama',
]

# Directories where GitHub Actions stages native assets
NATIVE_BINS_DIR = 'native_bins'
NATIVE_LIBS_DIR = 'native_libs'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_x11_library(name):
    """Check if a library name matches X11 patterns."""
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in X11_EXCLUDE_PATTERNS)

def filter_x11_binaries(binaries_list):
    """Remove X11 libraries from a binaries list (handles both 2-tuple and 3-tuple)."""
    filtered = []
    for item in binaries_list:
        if len(item) >= 2:
            name = item[0] if isinstance(item[0], str) else str(item[0])
            if not is_x11_library(name):
