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
                filtered.append(item)
    return filtered

def load_staged_binaries(directory, dest='.'):
    """
    Load native binaries from a staged directory.
    Returns list of (source_path, dest_folder) 2-tuples for Analysis().
    """
    binaries = []
    if os.path.isdir(directory):
        for filepath in glob.glob(os.path.join(directory, '*')):
            if os.path.isfile(filepath):
                filename = os.path.basename(filepath)
                if not is_x11_library(filename):
                    # 2-tuple format: (source_path, destination_folder)
                    binaries.append((filepath, dest))
    return binaries

def load_staged_dylibs(directory, dest='.'):
    """
    Load dylib files from a staged directory.
    Returns list of (source_path, dest_folder) 2-tuples for Analysis().
    """
    dylibs = []
    if os.path.isdir(directory):
        for filepath in glob.glob(os.path.join(directory, '*.dylib')):
            filename = os.path.basename(filepath)
            if not is_x11_library(filename):
                # 2-tuple format: (source_path, destination_folder)
                dylibs.append((filepath, dest))
    return dylibs

# =============================================================================
# COLLECT DEPENDENCIES
# =============================================================================

all_datas = [
    ('app.py', '.'),
]

# Add attached_assets if exists
if os.path.isdir('attached_assets'):
    all_datas.append(('attached_assets', 'attached_assets'))

all_binaries = []
all_hiddenimports = [
    # Streamlit internals
    'streamlit.runtime.scriptrunner.magic_funcs',
    'streamlit.web.bootstrap',
    # pywebview/pyobjc for macOS
    'webview.platforms.cocoa',
    'pyobjc',
    'objc',
    'Foundation',
    'AppKit',
    'WebKit',
    # Audio processing
    'mutagen.id3',
    'mutagen.mp3',
    'mutagen.wave',
    # scipy/numpy internals
    'scipy.signal',
    'scipy.fft',
    'scipy._lib.messagestream',
    'numpy.core._methods',
    'numpy.lib.format',
    # soundfile
    'soundfile',
    '_soundfile_data',
]

# Collect streamlit
try:
    datas, binaries, hiddenimports = collect_all('streamlit')
    all_datas += datas
    all_datas += copy_metadata('streamlit')
    all_binaries += filter_x11_binaries(binaries)
    all_hiddenimports += hiddenimports
except Exception as e:
    print(f"Warning: Could not collect streamlit: {e}")

# Collect audio processing libraries
audio_packages = ['librosa', 'pydub', 'pyloudnorm', 'pyrubberband', 'soundfile', 'mutagen']
for pkg in audio_packages:
    try:
        datas, binaries, hiddenimports = collect_all(pkg)
        all_datas += datas
        all_binaries += filter_x11_binaries(binaries)
        all_hiddenimports += hiddenimports
    except Exception as e:
        print(f"Warning: Could not collect {pkg}: {e}")

# Collect numpy and scipy
for pkg in ['numpy', 'scipy']:
    try:
        datas, binaries, hiddenimports = collect_all(pkg)
        all_datas += datas
        all_binaries += filter_x11_binaries(binaries)
        all_hiddenimports += hiddenimports
    except Exception as e:
        print(f"Warning: Could not collect {pkg}: {e}")

# Collect PIL/Pillow carefully (prone to X11 issues)
try:
    pillow_datas = collect_data_files('PIL')
    pillow_binaries = collect_dynamic_libs('PIL')
    all_datas += pillow_datas
    all_binaries += filter_x11_binaries(pillow_binaries)
    all_hiddenimports += ['PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont']
except Exception as e:
    print(f"Warning: Could not collect PIL: {e}")

# Collect pywebview
try:
    datas, binaries, hiddenimports = collect_all('webview')
    all_datas += datas
    all_binaries += filter_x11_binaries(binaries)
    all_hiddenimports += hiddenimports
except Exception as e:
    print(f"Warning: Could not collect webview: {e}")

# =============================================================================
# LOAD STAGED NATIVE BINARIES (from GitHub Actions)
# =============================================================================

# Load ffmpeg, ffprobe, rubberband executables (2-tuple format)
staged_bins = load_staged_binaries(NATIVE_BINS_DIR, '.')
all_binaries += staged_bins
print(f"Loaded {len(staged_bins)} staged binaries from {NATIVE_BINS_DIR}")

# Load native dylibs - codec libraries, etc. (2-tuple format)
staged_libs = load_staged_dylibs(NATIVE_LIBS_DIR, '.')
all_binaries += staged_libs
print(f"Loaded {len(staged_libs)} staged dylibs from {NATIVE_LIBS_DIR}")

# =============================================================================
# ANALYSIS
# =============================================================================

a = Analysis(
    ['run_raptur.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hook-runtime.py'],
    excludes=[
        'tkinter',
        'unittest',
        'xml',
        'pydoc',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Final X11 filtering pass on the internal TOC (which uses 3-tuple format)
a.binaries = [(name, path, typecode) for name, path, typecode in a.binaries 
              if not is_x11_library(name)]

# =============================================================================
# BUILD
# =============================================================================

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Raptur',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Raptur',
)

app = BUNDLE(
    coll,
    name='Raptur.app',
    icon=None,
    bundle_identifier='com.raptur.app',
    info_plist={
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSRequiresAquaSystemAppearance': False,
        'LSEnvironment': {
            'DYLD_LIBRARY_PATH': '@executable_path:@executable_path/../Frameworks',
        },
    },
)
