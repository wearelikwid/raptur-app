# -*- mode: python ; coding: utf-8 -*-
"""
Raptur PyInstaller Spec File
Handles macOS-specific bundling issues including:
- Filtering out X11 libraries (libxcb, libXau, libXdmcp) that cause Pillow crashes
- Collecting all audio processing dependencies
- Bundling native binaries (ffmpeg, rubberband)
- Proper pywebview/pyobjc integration
"""

import sys
import os

block_cipher = None

# Paths to native binaries (set by GitHub Actions)
native_bins_path = os.environ.get('NATIVE_BINS_PATH', 'native_bins')
native_libs_path = os.environ.get('NATIVE_LIBS_PATH', 'native_libs')

# Build binary list for native tools
binaries = []
if os.path.exists(native_bins_path):
    for binary in ['ffmpeg', 'ffprobe', 'rubberband']:
        binary_path = os.path.join(native_bins_path, binary)
        if os.path.exists(binary_path):
            binaries.append((binary_path, '.'))

# Add native libraries
if os.path.exists(native_libs_path):
    for lib in os.listdir(native_libs_path):
        if lib.endswith('.dylib'):
            binaries.append((os.path.join(native_libs_path, lib), '.'))

a = Analysis(
    ['run_raptur.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('app.py', '.'),
        ('attached_assets', 'attached_assets'),
    ],
    hiddenimports=[
        # Streamlit internals
        'streamlit.runtime.scriptrunner.magic_funcs',
        # pywebview/pyobjc for native window
        'webview.platforms.cocoa',
        'pyobjc',
        'objc',
        'Foundation',
        'AppKit',
        'WebKit',
        # mutagen for MP3 metadata
        'mutagen.id3',
        'mutagen.mp3',
        # scipy signal processing
        'scipy.signal',
        'scipy.fft',
        # numpy internals
        'numpy.core._methods',
        'numpy.lib.format',
        # librosa internals
        'librosa.util.decorators',
        # soundfile internals
        'soundfile',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# CRITICAL: Filter out X11 libraries that cause Pillow to crash on macOS
# These libraries (libxcb, libXau, libXdmcp) are only needed for X11 screenshot
# support which doesn't work on macOS anyway (macOS uses Quartz)
x11_libs_to_exclude = ['libxcb', 'libxau', 'libxdmcp', 'libx11']
a.binaries = [
    (name, path, type_) 
    for name, path, type_ in a.binaries 
    if not any(x in name.lower() for x in x11_libs_to_exclude)
]

# Collect all data and binaries from key packages
from PyInstaller.utils.hooks import collect_all, copy_metadata

# Streamlit
datas_streamlit, binaries_streamlit, hiddenimports_streamlit = collect_all('streamlit')
a.datas += datas_streamlit
a.binaries += binaries_streamlit
a.hiddenimports += hiddenimports_streamlit
a.datas += copy_metadata('streamlit')

# Audio libraries
for pkg in ['librosa', 'pydub', 'pyloudnorm', 'pyrubberband', 'soundfile', 'mutagen', 'scipy']:
    try:
        datas, binaries_pkg, hiddenimports = collect_all(pkg)
        a.datas += datas
        # Filter X11 libs from each package too
        a.binaries += [
            (n, p, t) for n, p, t in binaries_pkg
            if not any(x in n.lower() for x in x11_libs_to_exclude)
        ]
        a.hiddenimports += hiddenimports
    except Exception:
        pass

# NumPy and Pillow (with X11 filtering)
for pkg in ['numpy', 'Pillow']:
    try:
        datas, binaries_pkg, hiddenimports = collect_all(pkg)
        a.datas += datas
        # Filter X11 libs
        a.binaries += [
            (n, p, t) for n, p, t in binaries_pkg
            if not any(x in n.lower() for x in x11_libs_to_exclude)
        ]
        a.hiddenimports += hiddenimports
    except Exception:
        pass

# pywebview
try:
    datas_webview, binaries_webview, hiddenimports_webview = collect_all('webview')
    a.datas += datas_webview
    a.binaries += binaries_webview
    a.hiddenimports += hiddenimports_webview
except Exception:
    pass

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
    upx=False,  # UPX can break dylibs on macOS
    console=False,  # Windowed app
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
    },
)
