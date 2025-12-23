# -*- mode: python ; coding: utf-8 -*-
"""
Raptur PyInstaller Spec File
Handles macOS-specific bundling issues including:
- Filtering out X11 libraries (libxcb, libXau, libXdmcp) that cause Pillow crashes
- Collecting all audio processing dependencies
- Proper pywebview/pyobjc integration
"""

from PyInstaller.utils.hooks import collect_all, copy_metadata
import sys
import os

block_cipher = None

# X11 libraries to exclude (cause Pillow crashes on macOS)
x11_libs_to_exclude = ['libxcb', 'libxau', 'libxdmcp', 'libx11']

def filter_x11_binaries(binaries_list):
    """Filter out X11-related libraries that crash on macOS."""
    return [
        (name, path, typecode) 
        for name, path, typecode in binaries_list 
        if not any(x in name.lower() for x in x11_libs_to_exclude)
    ]

# Start with empty collections
all_datas = [
    ('app.py', '.'),
    ('attached_assets', 'attached_assets'),
]
all_binaries = []
all_hiddenimports = [
    'streamlit.runtime.scriptrunner.magic_funcs',
    'webview.platforms.cocoa',
    'pyobjc',
    'objc',
    'Foundation',
    'AppKit',
    'WebKit',
    'mutagen.id3',
    'mutagen.mp3',
    'scipy.signal',
    'scipy.fft',
    'numpy.core._methods',
    'numpy.lib.format',
    'soundfile',
]

# Collect streamlit with metadata
try:
    datas, binaries, hiddenimports = collect_all('streamlit')
    all_datas += datas
    all_datas += copy_metadata('streamlit')
    all_binaries += filter_x11_binaries(binaries)
    all_hiddenimports += hiddenimports
except Exception as e:
    print(f"Warning: Could not collect streamlit: {e}")

# Collect audio libraries
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

# Collect PIL/Pillow (filter X11 libs)
try:
    from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
    # Use individual collectors for Pillow since collect_all may fail
    pillow_datas = collect_data_files('PIL')
    pillow_binaries = collect_dynamic_libs('PIL')
    all_datas += pillow_datas
    all_binaries += filter_x11_binaries(pillow_binaries)
except Exception as e:
    print(f"Warning: Could not collect PIL: {e}")

# Collect pywebview
try:
    datas, binaries, hiddenimports = collect_all('webview')
    all_datas += datas
    all_binaries += binaries
    all_hiddenimports += hiddenimports
except Exception as e:
    print(f"Warning: Could not collect webview: {e}")

a = Analysis(
    ['run_raptur.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Additional X11 filtering on the final binaries list
a.binaries = filter_x11_binaries(a.binaries)

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
    },
)
