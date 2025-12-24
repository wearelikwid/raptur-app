import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import pyloudnorm as pyln
from pydub import AudioSegment
from scipy import signal
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2
import tempfile
import os
import io
import gc
import random
from pathlib import Path

st.set_page_config(page_title="Raptur", page_icon="🎵", layout="wide")

st.title("🎵 Raptur")
st.markdown("Upload - Mix - Download")

with st.sidebar:
    st.header("Get Raptur Desktop")
    st.markdown("Run Raptur locally on your Mac - no internet needed!")
    
    raptur_zip_path = "Raptur.zip"
    if os.path.exists(raptur_zip_path):
        with open(raptur_zip_path, "rb") as zip_file:
            st.download_button(
                label="Download Raptur for macOS",
                data=zip_file,
                file_name="Raptur.zip",
                mime="application/zip",
                use_container_width=True
            )
        st.caption("Unzip and double-click Raptur.app to run")
    else:
        st.info("Desktop version coming soon!")
    
    st.divider()

TARGET_LUFS = -14.0
ANALYSIS_SR = 22050  # Lower sample rate for BPM/key/HPSS analysis (4x faster than 44100)

CAMELOT_MAPPING = {
    'C': '8B', 'C#': '3B', 'Db': '3B', 'D': '10B', 'D#': '5B', 'Eb': '5B',
    'E': '12B', 'F': '7B', 'F#': '2B', 'Gb': '2B', 'G': '9B', 'G#': '4B',
    'Ab': '4B', 'A': '11B', 'A#': '6B', 'Bb': '6B', 'B': '1B',
    'Cm': '5A', 'C#m': '12A', 'Dbm': '12A', 'Dm': '7A', 'D#m': '2A', 'Ebm': '2A',
    'Em': '9A', 'Fm': '4A', 'F#m': '11A', 'Gbm': '11A', 'Gm': '6A', 'G#m': '1A',
    'Abm': '1A', 'Am': '8A', 'A#m': '3A', 'Bbm': '3A', 'Bm': '10A'
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def filter_bpm_outliers(uploaded_files, tolerance=0.15):
    """Quick BPM scan to detect and filter outliers BEFORE heavy processing.
    
    This is a fast "gatekeeper" step that:
    1. Loads only first 30 seconds of each track (lightweight)
    2. Detects BPM (raw value, no halftime/doubletime correction)
    3. Uses median (not mean) to avoid outlier influence
    4. Filters tracks outside ±tolerance of median
    
    Args:
        uploaded_files: List of uploaded file objects
        tolerance: Fraction of median to use as threshold (default 0.15 = ±15%)
    
    Returns:
        valid_files: List of files within BPM tolerance
        rejected_files: List of (filename, bpm) tuples for rejected tracks
        bpm_data: Dict mapping filename to detected BPM (for reuse)
    """
    bpm_list = []
    file_bpm_pairs = []
    
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1], delete=False) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            y, sr = librosa.load(tmp_path, duration=30, mono=True)
            os.unlink(tmp_path)
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0])
            tempo = float(tempo)
            
            if tempo <= 0 or tempo > 300:
                tempo = 120.0
            
            bpm_list.append(tempo)
            file_bpm_pairs.append((uploaded_file, tempo))
            
            del y
            gc.collect()
            
        except Exception as e:
            bpm_list.append(120.0)
            file_bpm_pairs.append((uploaded_file, 120.0))
    
    if len(bpm_list) < 2:
        return uploaded_files, [], {f.name: bpm for f, bpm in file_bpm_pairs}
    
    median_bpm = np.median(bpm_list)
    lower_bound = median_bpm * (1 - tolerance)
    upper_bound = median_bpm * (1 + tolerance)
    
    valid_files = []
    rejected_files = []
    bpm_data = {}
    
    for uploaded_file, bpm in file_bpm_pairs:
        bpm_data[uploaded_file.name] = bpm
        if lower_bound <= bpm <= upper_bound:
            uploaded_file.seek(0)
            valid_files.append(uploaded_file)
        else:
            rejected_files.append((uploaded_file.name, bpm))
    
    return valid_files, rejected_files, bpm_data

def detect_key(y, sr):
    """Detect musical key using chroma features and return Camelot code.
    
    Uses Krumhansl-Schmuckler key-finding algorithm:
    1. Extract chroma features (pitch class distribution)
    2. Correlate against major/minor key profiles
    3. Return best matching key in Camelot notation
    
    OPTIMIZATION: Only analyzes first 15 seconds at ANALYSIS_SR (22050Hz)
    """
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    
    # Limit to first 15 seconds for speed (key doesn't change mid-song)
    max_samples = int(15 * sr)
    if len(y_mono) > max_samples:
        y_mono = y_mono[:max_samples]
    
    # Downsample to ANALYSIS_SR for faster chroma extraction
    if sr != ANALYSIS_SR:
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        analysis_sr = ANALYSIS_SR
    else:
        analysis_sr = sr
    
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=analysis_sr)
    chroma_sum = np.sum(chroma, axis=1)
    chroma_sum = chroma_sum / (np.sum(chroma_sum) + 1e-8)
    
    best_corr = -np.inf
    best_key = 'C'
    
    for i in range(12):
        major_rotated = np.roll(MAJOR_PROFILE, i)
        major_rotated = major_rotated / (np.sum(major_rotated) + 1e-8)
        major_corr = np.corrcoef(chroma_sum, major_rotated)[0, 1]
        
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = NOTE_NAMES[i]
        
        minor_rotated = np.roll(MINOR_PROFILE, i)
        minor_rotated = minor_rotated / (np.sum(minor_rotated) + 1e-8)
        minor_corr = np.corrcoef(chroma_sum, minor_rotated)[0, 1]
        
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = NOTE_NAMES[i] + 'm'
    
    camelot = CAMELOT_MAPPING.get(best_key, '8B')
    return camelot

def get_camelot_number(camelot_code):
    """Extract the numeric part of a Camelot code for sorting."""
    if not camelot_code:
        return 0
    num_str = ''.join(c for c in camelot_code if c.isdigit())
    return int(num_str) if num_str else 0

def load_and_sanitize(file_path, top_db=20):
    """Load audio file, trim silence, quantize to beat grid, and normalize to -14 LUFS.
    
    Grid-locking process:
    1. Trim leading/trailing silence
    2. Detect beats and snap to first kick (the "One")
    3. Trim to exact integer number of beats (prevents phase drift)
    4. Normalize to -14 LUFS
    
    Also captures intro (before first kick) and tail (after grid-lock) segments
    for seamless mix boundaries.
    
    OPTIMIZATION: Uses downsampled audio (22050Hz) for beat detection - 4x faster
    while maintaining accuracy since BPM detection works well at lower sample rates.
    """
    y, sr = librosa.load(file_path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.array([y, y])
    
    y_mono = librosa.to_mono(y)
    _, index = librosa.effects.trim(y_mono, top_db=top_db)
    y_trimmed = y[:, index[0]:index[1]]
    
    y_mono_trimmed = librosa.to_mono(y_trimmed)
    
    # Use downsampled audio for beat tracking (4x faster)
    if sr != ANALYSIS_SR:
        y_mono_analysis = librosa.resample(y_mono_trimmed, orig_sr=sr, target_sr=ANALYSIS_SR)
        analysis_sr = ANALYSIS_SR
    else:
        y_mono_analysis = y_mono_trimmed
        analysis_sr = sr
    
    tempo, beat_frames_analysis = librosa.beat.beat_track(y=y_mono_analysis, sr=analysis_sr)
    
    # Convert beat frames from analysis sample rate back to original sr
    beat_frames = librosa.core.frames_to_samples(beat_frames_analysis, hop_length=512)
    beat_frames = (beat_frames * sr / analysis_sr).astype(int)
    beat_frames = librosa.core.samples_to_frames(beat_frames, hop_length=512)
    
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    tempo = float(tempo)
    if tempo <= 0 or tempo > 300:
        tempo = 120.0
    
    intro_segment = np.array([[], []])
    tail_segment = np.array([[], []])
    
    if len(beat_frames) > 1:
        beat_samples = librosa.frames_to_samples(beat_frames)
        first_beat_sample = beat_samples[0]
        
        intro_segment = y_trimmed[:, :first_beat_sample]
        
        y_trimmed = y_trimmed[:, first_beat_sample:]
        
        samples_per_beat = sr * 60.0 / tempo
        total_samples = y_trimmed.shape[1]
        num_full_beats = int(total_samples / samples_per_beat)
        
        if num_full_beats > 0:
            valid_samples = int(num_full_beats * samples_per_beat)
            
            tail_segment = y_trimmed[:, valid_samples:]
            
            y_trimmed = y_trimmed[:, :valid_samples]
    
    meter = pyln.Meter(sr)
    y_for_loudness = y_trimmed.T if y_trimmed.ndim > 1 else y_trimmed
    
    gain_linear = 1.0
    try:
        loudness = meter.integrated_loudness(y_for_loudness)
        if np.isfinite(loudness) and loudness < 0:
            gain_db = TARGET_LUFS - loudness
            gain_linear = 10 ** (gain_db / 20.0)
            y_normalized = y_trimmed * gain_linear
            max_val = np.max(np.abs(y_normalized))
            if max_val > 0.99:
                limiter_gain = 0.95 / max_val
                y_normalized = y_normalized * limiter_gain
                gain_linear = gain_linear * limiter_gain
        else:
            y_normalized = y_trimmed
    except Exception:
        y_normalized = y_trimmed
    
    if intro_segment.shape[1] > 0:
        intro_segment = intro_segment * gain_linear
    if tail_segment.shape[1] > 0:
        tail_segment = tail_segment * gain_linear
    
    key = detect_key(y_normalized, sr)
    
    intro_safe_sec, outro_safe_sec = analyze_structural_safe_zones(y_normalized, sr)
    
    return y_normalized, sr, tempo, key, intro_safe_sec, outro_safe_sec, intro_segment, tail_segment

def analyze_structural_safe_zones(y, sr, vocal_threshold=0.15, window_sec=1.0):
    """Analyze audio to find intro/outro safe zones (drum-only sections without vocals).
    
    Uses Harmonic-Percussive Source Separation (HPSS) to isolate harmonic content
    (vocals/melodies), then scans RMS energy to find where vocals start/end.
    
    OPTIMIZATION: Downsamples to ANALYSIS_SR (22050Hz) before HPSS - 4x faster
    
    Args:
        y: Audio array (stereo [2, samples] or mono)
        sr: Sample rate
        vocal_threshold: RMS threshold above which vocals/melody are detected (0.0-1.0)
        window_sec: Window size in seconds for RMS calculation
    
    Returns:
        tuple: (intro_safe_zone_sec, outro_safe_zone_sec)
            - intro_safe_zone_sec: Duration in seconds before vocals start
            - outro_safe_zone_sec: Duration in seconds after vocals end
    """
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    
    # Downsample to ANALYSIS_SR for faster HPSS (we only need structure, not detail)
    if sr != ANALYSIS_SR:
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        analysis_sr = ANALYSIS_SR
    else:
        analysis_sr = sr
    
    y_harm, y_perc = librosa.effects.hpss(y_mono)
    
    hop_length = int(analysis_sr * window_sec)
    frame_length = hop_length
    
    rms = librosa.feature.rms(y=y_harm, frame_length=frame_length, hop_length=hop_length)[0]
    
    if len(rms) == 0:
        return 0.0, 0.0
    
    rms_max = np.max(rms)
    if rms_max > 0:
        rms_normalized = rms / rms_max
    else:
        return 0.0, 0.0
    
    intro_safe_sec = 0.0
    for i, energy in enumerate(rms_normalized):
        if energy > vocal_threshold:
            intro_safe_sec = i * window_sec
            break
    else:
        intro_safe_sec = len(rms_normalized) * window_sec
    
    outro_safe_sec = 0.0
    for i in range(len(rms_normalized) - 1, -1, -1):
        if rms_normalized[i] > vocal_threshold:
            outro_safe_sec = (len(rms_normalized) - 1 - i) * window_sec
            break
    else:
        outro_safe_sec = len(rms_normalized) * window_sec
    
    total_duration = len(y_mono) / analysis_sr
    intro_safe_sec = min(intro_safe_sec, total_duration)
    outro_safe_sec = min(outro_safe_sec, total_duration)
    
    return intro_safe_sec, outro_safe_sec

def analyze_bpm(y, sr):
    """Analyze BPM using librosa beat tracking."""
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    tempo = float(tempo)
    if tempo <= 0 or tempo > 300:
        tempo = 120.0
    return tempo

def time_stretch_audio(y, sr, original_bpm, target_bpm):
    """Time stretch audio to match target BPM using rubberband."""
    if original_bpm <= 0 or target_bpm <= 0:
        return y
    if abs(original_bpm - target_bpm) < 0.5:
        return y
    stretch_ratio = target_bpm / original_bpm
    if stretch_ratio < 0.5 or stretch_ratio > 2.0:
        return y
    if y.ndim == 1:
        y_stretched = pyrb.time_stretch(y, sr, stretch_ratio)
    else:
        channels = []
        for i in range(y.shape[0]):
            stretched = pyrb.time_stretch(y[i], sr, stretch_ratio)
            channels.append(stretched)
        y_stretched = np.array(channels)
    return y_stretched

def calculate_beat_samples(bpm, sr, num_beats):
    """Calculate number of samples for given number of beats."""
    seconds_per_beat = 60.0 / bpm
    samples = int(seconds_per_beat * num_beats * sr)
    return samples

def butter_highpass_filter(data, cutoff, fs, order=2):
    """Apply a Butterworth high-pass filter to remove bass frequencies."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data, axis=1)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=2):
    """Apply a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data, axis=1)
    return y

def apply_riser_filter(audio, sr, riser_samples, start_freq=20, end_freq=400, steps=64):
    """Apply a gradual high-pass filter sweep (riser effect) to audio in-place.
    
    This creates a natural DJ-style build-up by gradually removing bass frequencies
    over the specified duration. The effect ramps smoothly from start_freq to end_freq.
    
    CRITICAL: This function preserves array length - no timing/grid changes.
    
    Args:
        audio: Stereo array [2, samples] - the audio to process
        sr: Sample rate
        riser_samples: Number of samples over which to apply the sweep
        start_freq: Starting high-pass frequency in Hz (default 20Hz = nearly full bass)
        end_freq: Ending high-pass frequency in Hz (default 400Hz = bass removed)
        steps: Number of filter steps for smooth transition
    
    Returns:
        Processed audio with same shape as input
    """
    if audio.ndim == 1:
        audio = np.array([audio, audio])
    
    num_samples = audio.shape[1]
    if riser_samples > num_samples:
        riser_samples = num_samples
    
    riser_start = num_samples - riser_samples
    
    result = audio.copy().astype(np.float64)
    
    riser_region = result[:, riser_start:]
    
    nyq = 0.5 * sr
    samples_per_step = riser_samples // steps
    
    processed_chunks = []
    
    for i in range(steps):
        start_idx = i * samples_per_step
        end_idx = (i + 1) * samples_per_step if i < steps - 1 else riser_samples
        
        chunk = riser_region[:, start_idx:end_idx]
        
        progress = i / (steps - 1)
        current_cutoff = start_freq + (end_freq - start_freq) * progress
        
        normal_cutoff = current_cutoff / nyq
        normal_cutoff = max(0.001, min(0.99, normal_cutoff))
        
        if chunk.shape[1] > 12:
            b, a = signal.butter(2, normal_cutoff, btype='high', analog=False)
            filtered_chunk = signal.filtfilt(b, a, chunk, axis=1)
        else:
            filtered_chunk = chunk
        
        processed_chunks.append(filtered_chunk)
    
    processed_riser = np.concatenate(processed_chunks, axis=1)
    
    result[:, riser_start:] = processed_riser
    
    return result.astype(np.float32)

def apply_dynamic_sweep(data, sr, start_freq=200, end_freq=5000, steps=32):
    """Apply a time-varying high-pass filter sweep for a 'riser' effect.
    
    Simulates a DJ gradually turning the filter knob by splitting audio into
    chunks and applying progressively higher cutoff frequencies using
    logarithmic interpolation (sounds more natural than linear).
    
    Args:
        data: Stereo audio array [2, samples]
        sr: Sample rate
        start_freq: Starting cutoff frequency (Hz)
        end_freq: Ending cutoff frequency (Hz)  
        steps: Number of filter chunks (more = smoother but slower)
    
    Returns:
        Filtered audio with gradual frequency sweep
    """
    if data.ndim == 1:
        data = np.array([data, data])
    
    total_samples = data.shape[1]
    chunk_size = total_samples // steps
    
    if chunk_size < 1:
        return data
    
    processed_chunks = []
    nyq = 0.5 * sr
    
    for i in range(steps):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < steps - 1 else total_samples
        
        chunk = data[:, start_idx:end_idx]
        
        current_cutoff = start_freq * ((end_freq / start_freq) ** (i / (steps - 1)))
        
        normal_cutoff = current_cutoff / nyq
        if normal_cutoff >= 1.0:
            normal_cutoff = 0.99
        if normal_cutoff <= 0:
            normal_cutoff = 0.01
        
        if chunk.shape[1] > 12:
            b, a = signal.butter(2, normal_cutoff, btype='high', analog=False)
            filtered_chunk = signal.filtfilt(b, a, chunk, axis=1)
        else:
            filtered_chunk = chunk
        
        processed_chunks.append(filtered_chunk)
    
    return np.concatenate(processed_chunks, axis=1)

def fine_tune_alignment(track_a_segment, track_b_segment, sr):
    """Phase-lock two audio segments by cross-correlating their kick drums.
    
    Uses sub-beat alignment via signal processing:
    1. Low-pass filter at 100Hz to isolate kicks
    2. Cross-correlate to find optimal alignment
    3. Limit shift to ±100ms to prevent wrong-beat alignment
    4. Return the offset so the caller can adjust the remainder slice
    
    Args:
        track_a_segment: Stereo array [2, samples] - end of outgoing track
        track_b_segment: Stereo array [2, samples] - start of incoming track
        sr: Sample rate
    
    Returns:
        tuple: (aligned segment, offset in samples)
            - offset > 0 means track_b should start later (skip offset samples from remainder)
            - offset < 0 means track_b should start earlier (already included in overlap)
    """
    max_shift_samples = int(0.1 * sr)
    
    a_mono = np.mean(track_a_segment, axis=0) if track_a_segment.ndim > 1 else track_a_segment
    b_mono = np.mean(track_b_segment, axis=0) if track_b_segment.ndim > 1 else track_b_segment
    
    nyq = 0.5 * sr
    cutoff = 100 / nyq
    if cutoff >= 1.0:
        cutoff = 0.99
    b_coef, a_coef = signal.butter(2, cutoff, btype='low', analog=False)
    
    a_low = signal.filtfilt(b_coef, a_coef, a_mono)
    b_low = signal.filtfilt(b_coef, a_coef, b_mono)
    
    correlation = signal.correlate(a_low, b_low, mode='full')
    
    mid_point = len(correlation) // 2
    search_start = max(0, mid_point - max_shift_samples)
    search_end = min(len(correlation), mid_point + max_shift_samples)
    
    search_region = correlation[search_start:search_end]
    local_peak_idx = np.argmax(search_region)
    
    offset = local_peak_idx - (mid_point - search_start)
    
    if abs(offset) > max_shift_samples:
        return track_b_segment, 0
    
    if offset == 0:
        return track_b_segment, 0
    
    return track_b_segment, offset

def apply_hpss_drum_swap(audio_a, audio_b, overlap_len, sr):
    """Applies HPSS-based "drum swap" transition for forced overlaps.
    
    Used when safe zones are too short - we force a long transition but use
    HPSS to surgically separate Track A's stems so vocals don't clash.
    
    Track A: Drums (percussive) fade out FAST, Vocals (harmonic) fade out SLOW
    Track B: Full audio fades in normally
    
    This creates a "mashup" style transition where Track A's vocals float over
    Track B's full mix before disappearing.
    """
    if audio_a.ndim == 1:
        audio_a = np.array([audio_a, audio_a])
    if audio_b.ndim == 1:
        audio_b = np.array([audio_b, audio_b])
    
    a_left_harm, a_left_perc = librosa.effects.hpss(audio_a[0].astype(np.float64))
    a_right_harm, a_right_perc = librosa.effects.hpss(audio_a[1].astype(np.float64))
    
    perc_fade = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32) ** 2
    
    harm_fade = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32) ** 0.5
    
    fade_in = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)
    
    out_a_left = (a_left_harm * harm_fade + a_left_perc * perc_fade).astype(np.float32)
    out_a_right = (a_right_harm * harm_fade + a_right_perc * perc_fade).astype(np.float32)
    out_a = np.array([out_a_left, out_a_right])
    
    out_b = (audio_b.astype(np.float64) * fade_in).astype(np.float32)
    
    return out_a + out_b

def apply_fx_transition(audio_a, audio_b, overlap_len, sr, fx_type="riser"):
    """Applies DJ transition effect based on fx_type.
    
    Args:
        audio_a: Outgoing track segment (stereo [2, samples])
        audio_b: Incoming track segment (stereo [2, samples])
        overlap_len: Number of samples in the overlap
        sr: Sample rate
        fx_type: "riser" for natural transitions, "drum_swap" for forced overlaps
    
    Returns:
        Blended audio segment
    """
    if fx_type == "drum_swap":
        return apply_hpss_drum_swap(audio_a, audio_b, overlap_len, sr)
    
    fade_out = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)
    
    out_a = audio_a.copy().astype(np.float64)
    out_b = audio_b.copy().astype(np.float64)
    
    out_a = apply_dynamic_sweep(out_a, sr, start_freq=20, end_freq=400, steps=32)
    
    out_a = (out_a * fade_out).astype(np.float32)
    out_b = (out_b * fade_in).astype(np.float32)
    
    return out_a + out_b

def process_uploaded_files(uploaded_files, progress_bar, status_text):
    """Process uploaded files: save to temp, trim silence, normalize, analyze BPM.
    Returns metadata with file paths instead of audio arrays to save memory.
    
    OPTIMIZATION: Uses downsampled audio (22050Hz) for BPM/key/HPSS analysis - 4x faster.
    """
    tracks_metadata = []
    temp_dir = tempfile.mkdtemp()
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files) * 0.5
        progress_bar.progress(progress)
        status_text.text(f"Processing: {uploaded_file.name}")
        
        input_path = os.path.join(temp_dir, f"input_{i}_{uploaded_file.name}")
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            status_text.text(f"Processing & grid-locking: {uploaded_file.name}")
            y_sanitized, sr, bpm, key, intro_safe_sec, outro_safe_sec, intro_segment, tail_segment = load_and_sanitize(input_path)
            
            duration = y_sanitized.shape[1] / sr if y_sanitized.ndim > 1 else len(y_sanitized) / sr
            
            sanitized_path = os.path.join(temp_dir, f"sanitized_{i}.wav")
            sf.write(sanitized_path, y_sanitized.T, sr)
            
            intro_path = None
            if intro_segment.shape[1] > 0:
                intro_path = os.path.join(temp_dir, f"intro_{i}.wav")
                sf.write(intro_path, intro_segment.T, sr)
            
            tail_path = None
            if tail_segment.shape[1] > 0:
                tail_path = os.path.join(temp_dir, f"tail_{i}.wav")
                sf.write(tail_path, tail_segment.T, sr)
            
            del y_sanitized, intro_segment, tail_segment
            gc.collect()
            
            os.remove(input_path)
            
            tracks_metadata.append({
                'name': uploaded_file.name,
                'file_path': sanitized_path,
                'sr': sr,
                'bpm': bpm,
                'key': key,
                'duration': duration,
                'intro_safe_sec': intro_safe_sec,
                'outro_safe_sec': outro_safe_sec,
                'intro_path': intro_path,
                'tail_path': tail_path
            })
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            if os.path.exists(input_path):
                os.remove(input_path)
            continue
    
    return tracks_metadata, temp_dir

def calculate_dynamic_transition_beats(track_a_outro_sec, track_b_intro_sec, bpm, min_beats=4, max_beats=32):
    """Calculate dynamic transition length based on available safe zones with Grid-Snap.
    
    CRITICAL: All transitions must be multiples of 4 beats to preserve beatmatching.
    
    Args:
        track_a_outro_sec: Outro safe zone duration (seconds) from Track A
        track_b_intro_sec: Intro safe zone duration (seconds) from Track B
        bpm: Target BPM for the mix
        min_beats: Minimum transition length (default 4 beats = 1 bar)
        max_beats: Maximum transition length (default 32 beats = 8 bars)
    
    Returns:
        int: Number of beats for transition (always multiple of 4)
    """
    safe_duration_sec = min(track_a_outro_sec, track_b_intro_sec)
    
    beats_available = safe_duration_sec * (bpm / 60.0)
    
    valid_beats = int(beats_available) - (int(beats_available) % 4)
    
    if valid_beats < min_beats:
        valid_beats = min_beats
    
    if valid_beats > max_beats:
        valid_beats = max_beats
    
    return valid_beats

def create_streaming_mix(tracks_metadata, target_bpm, progress_callback=None):
    """Create a continuous mix with content-aware dynamic transitions.
    
    Uses Smart Phrasing: analyzes where vocals start/end in each track and
    dynamically resizes transitions to fit into "safe zones" (drum-only sections).
    All transition lengths are Grid-Snapped to multiples of 4 beats.
    
    Args:
        tracks_metadata: List of track metadata dicts (must include intro_safe_sec, outro_safe_sec)
        target_bpm: Target BPM for the mix
        progress_callback: Optional callback for progress updates
    """
    if not tracks_metadata:
        return None, None
    
    sr = tracks_metadata[0]['sr']
    
    mixed_left = np.array([], dtype=np.float32)
    mixed_right = np.array([], dtype=np.float32)
    
    transition_log = []
    
    for i, track_meta in enumerate(tracks_metadata):
        y, file_sr = sf.read(track_meta['file_path'])
        y = y.T
        
        if y.ndim == 1:
            y = np.array([y, y])
        
        original_bpm = track_meta['bpm']
        y_stretched = time_stretch_audio(y, sr, original_bpm, target_bpm)
        
        del y
        gc.collect()
        
        if y_stretched.ndim == 1:
            y_stretched = np.array([y_stretched, y_stretched])
        
        if i == 0:
            intro_path = track_meta.get('intro_path')
            if intro_path and os.path.exists(intro_path):
                intro_audio, _ = sf.read(intro_path)
                intro_audio = intro_audio.T
                if intro_audio.ndim == 1:
                    intro_audio = np.array([intro_audio, intro_audio])
                intro_stretched = time_stretch_audio(intro_audio, sr, original_bpm, target_bpm)
                if intro_stretched.ndim == 1:
                    intro_stretched = np.array([intro_stretched, intro_stretched])
                mixed_left = np.concatenate([intro_stretched[0].astype(np.float32), y_stretched[0].astype(np.float32)])
                mixed_right = np.concatenate([intro_stretched[1].astype(np.float32), y_stretched[1].astype(np.float32)])
                del intro_audio, intro_stretched
            else:
                mixed_left = y_stretched[0].astype(np.float32)
                mixed_right = y_stretched[1].astype(np.float32)
            if progress_callback:
                progress_callback(f"Mixing track {i+1}/{len(tracks_metadata)}: {track_meta['name']}")
        else:
            prev_track = tracks_metadata[i - 1]
            track_a_outro = prev_track.get('outro_safe_sec', 8.0)
            track_b_intro = track_meta.get('intro_safe_sec', 8.0)
            
            safe_duration_sec = min(track_a_outro, track_b_intro)
            raw_beats = safe_duration_sec * (target_bpm / 60.0)
            natural_beats = int(raw_beats) - (int(raw_beats) % 4)
            
            if natural_beats >= 16:
                transition_beats = min(natural_beats, 32)
                fx_type = "riser"
                transition_mode = "natural"
            else:
                transition_beats = 32
                fx_type = "drum_swap"
                transition_mode = "forced"
            
            transition_samples = calculate_beat_samples(target_bpm, sr, transition_beats)
            
            if transition_mode == "natural":
                transition_log.append(f"{track_meta['name']}: {transition_beats}-beat riser (A outro={track_a_outro:.1f}s, B intro={track_b_intro:.1f}s)")
            else:
                transition_log.append(f"{track_meta['name']}: {transition_beats}-beat DRUM SWAP [forced] (safe zone only {natural_beats} beats)")
            
            if progress_callback:
                progress_callback(f"Mixing {track_meta['name']} ({transition_beats}-beat {transition_mode})...")
            
            overlap_samples = min(transition_samples, len(mixed_left), len(y_stretched[0]))
            
            segment_a = np.array([
                mixed_left[-overlap_samples:], 
                mixed_right[-overlap_samples:]
            ])
            
            segment_b = np.array([
                y_stretched[0][:overlap_samples], 
                y_stretched[1][:overlap_samples]
            ])
            
            _, phase_offset = fine_tune_alignment(segment_a, segment_b, sr)
            
            if phase_offset > 0:
                pad_amount = phase_offset
                segment_b = np.concatenate([
                    np.zeros((2, pad_amount)),
                    y_stretched[:, :overlap_samples - pad_amount]
                ], axis=1)
                remainder_start = overlap_samples - pad_amount
                    
            elif phase_offset < 0:
                skip_amount = abs(phase_offset)
                end_idx = overlap_samples + skip_amount
                available = y_stretched.shape[1]
                
                if end_idx <= available:
                    segment_b = np.array([
                        y_stretched[0][skip_amount:end_idx], 
                        y_stretched[1][skip_amount:end_idx]
                    ])
                    remainder_start = end_idx
                else:
                    actual_slice = y_stretched[:, skip_amount:]
                    pad_needed = overlap_samples - actual_slice.shape[1]
                    if pad_needed > 0:
                        segment_b = np.concatenate([
                            actual_slice,
                            np.zeros((2, pad_needed))
                        ], axis=1)
                    else:
                        segment_b = actual_slice[:, :overlap_samples]
                    remainder_start = available
            else:
                remainder_start = overlap_samples
            
            remainder_start = max(0, min(remainder_start, y_stretched.shape[1]))
            
            mixed_segment = apply_fx_transition(segment_a, segment_b, overlap_samples, sr, fx_type=fx_type)
            
            mixed_left = mixed_left[:-overlap_samples]
            mixed_right = mixed_right[:-overlap_samples]
            
            mixed_left = np.concatenate([mixed_left, mixed_segment[0]])
            mixed_right = np.concatenate([mixed_right, mixed_segment[1]])
            
            remainder_left = y_stretched[0][remainder_start:].astype(np.float32)
            remainder_right = y_stretched[1][remainder_start:].astype(np.float32)
            
            mixed_left = np.concatenate([mixed_left, remainder_left])
            mixed_right = np.concatenate([mixed_right, remainder_right])
        
        del y_stretched
        gc.collect()
    
    if len(tracks_metadata) > 0:
        last_track = tracks_metadata[-1]
        tail_path = last_track.get('tail_path')
        if tail_path and os.path.exists(tail_path):
            tail_audio, _ = sf.read(tail_path)
            tail_audio = tail_audio.T
            if tail_audio.ndim == 1:
                tail_audio = np.array([tail_audio, tail_audio])
            tail_stretched = time_stretch_audio(tail_audio, sr, last_track['bpm'], target_bpm)
            if tail_stretched.ndim == 1:
                tail_stretched = np.array([tail_stretched, tail_stretched])
            mixed_left = np.concatenate([mixed_left, tail_stretched[0].astype(np.float32)])
            mixed_right = np.concatenate([mixed_right, tail_stretched[1].astype(np.float32)])
            del tail_audio, tail_stretched
            gc.collect()
    
    final_audio = np.array([mixed_left, mixed_right])
    
    max_val = np.max(np.abs(final_audio))
    if max_val > 0.99:
        final_audio = final_audio * 0.95 / max_val
    
    return final_audio, sr, transition_log

def cleanup_temp_files(tracks_metadata, temp_dir):
    """Clean up temporary files including intro/tail segments."""
    for track in tracks_metadata:
        if 'file_path' in track and os.path.exists(track['file_path']):
            os.remove(track['file_path'])
        if 'intro_path' in track and track['intro_path'] and os.path.exists(track['intro_path']):
            os.remove(track['intro_path'])
        if 'tail_path' in track and track['tail_path'] and os.path.exists(track['tail_path']):
            os.remove(track['tail_path'])
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

uploaded_files = st.file_uploader(
    "Upload MP3/WAV files",
    type=['mp3', 'wav'],
    accept_multiple_files=True,
    help="Select multiple audio files to create a continuous mix"
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
    
    st.subheader("Mix Details")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mix_name = st.text_input(
            "Mix Name",
            value="My Mix",
            help="Name for your mix (used in filename and metadata)"
        )
    
    with col2:
        artwork_file = st.file_uploader(
            "Album Artwork (optional)",
            type=['jpg', 'jpeg', 'png'],
            help="Cover art to embed in your MP3"
        )
        if artwork_file:
            try:
                artwork_file.seek(0)
                artwork_image = Image.open(artwork_file)
                st.image(artwork_image, width=100, caption="Preview")
            except Exception:
                st.warning("Could not preview image")
    
    if st.button("🎛️ Generate Mix", type="primary", use_container_width=True):
        if len(uploaded_files) < 2:
            st.warning("Please upload at least 2 files to create a mix.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("⚡ Quick BPM scan for outliers...")
            valid_files, rejected_files, bpm_data = filter_bpm_outliers(uploaded_files)
            
            if rejected_files:
                rejected_list = ", ".join([f"{name} ({bpm:.1f} BPM)" for name, bpm in rejected_files])
                median_bpm = np.median([bpm for _, bpm in [(f.name, bpm_data[f.name]) for f in valid_files]] if valid_files else [120])
                st.warning(f"⚠️ Skipped BPM outliers (outside ±15% of {median_bpm:.0f} BPM median): {rejected_list}")
            
            if len(valid_files) < 2:
                st.error("Not enough tracks remaining after filtering outliers. Need at least 2 compatible tracks.")
            else:
                temp_dir = None
                tracks_metadata = []
                
                try:
                    with st.spinner("Processing files..."):
                        status_text.text("Loading, trimming silence, and normalizing volume...")
                        tracks_metadata, temp_dir = process_uploaded_files(valid_files, progress_bar, status_text)
                    
                    if len(tracks_metadata) >= 2:
                        status_text.text("Sorting tracks by BPM and Key...")
                        tracks_metadata.sort(key=lambda x: (round(x['bpm']), get_camelot_number(x['key'])))
                        
                        st.subheader("📊 Track Analysis (Sorted by BPM & Key)")
                        for i, track in enumerate(tracks_metadata):
                            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                            with col1:
                                st.write(f"**{i+1}. {track['name']}**")
                            with col2:
                                st.write(f"🎵 {track['bpm']:.1f} BPM")
                            with col3:
                                st.write(f"🎹 {track['key']}")
                            with col4:
                                st.write(f"⏱️ {track['duration']:.1f}s")
                        
                        average_bpm = sum(t['bpm'] for t in tracks_metadata) / len(tracks_metadata)
                        st.info(f"🎯 Target BPM (Average): {average_bpm:.1f}")
                        
                        status_text.text("Creating beatmatched mix (sequential processing)...")
                        progress_bar.progress(0.6)
                        
                        def update_progress(msg):
                            status_text.text(msg)
                        
                        result = create_streaming_mix(
                            tracks_metadata, 
                            average_bpm, 
                            progress_callback=update_progress
                        )
                        
                        if result is not None and result[0] is not None:
                            mixed_audio, sr, transition_log = result
                            progress_bar.progress(0.9)
                            status_text.text("Exporting mix...")
                            
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                                tmp_wav_path = tmp_wav.name
                                sf.write(tmp_wav_path, mixed_audio.T, sr)
                            
                            del mixed_audio
                            gc.collect()
                            
                            audio_segment = AudioSegment.from_wav(tmp_wav_path)
                            
                            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                                tmp_mp3_path = tmp_mp3.name
                            
                            audio_segment.export(tmp_mp3_path, format='mp3', bitrate='320k')
                            
                            del audio_segment
                            gc.collect()
                            
                            os.unlink(tmp_wav_path)
                            
                            audio = MP3(tmp_mp3_path, ID3=ID3)
                            try:
                                audio.add_tags()
                            except:
                                pass
                            
                            safe_mix_name = mix_name.strip() if mix_name else "My Mix"
                            audio.tags.add(TIT2(encoding=3, text=safe_mix_name))
                            
                            if artwork_file:
                                artwork_file.seek(0)
                                artwork_data = artwork_file.read()
                                mime_type = 'image/jpeg' if artwork_file.name.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                                audio.tags.add(APIC(
                                    encoding=3,
                                    mime=mime_type,
                                    type=3,
                                    desc='Cover',
                                    data=artwork_data
                                ))
                            
                            audio.save()
                            
                            with open(tmp_mp3_path, 'rb') as f:
                                mp3_buffer = io.BytesIO(f.read())
                            
                            os.unlink(tmp_mp3_path)
                            
                            progress_bar.progress(1.0)
                            status_text.text("Mix complete!")
                            
                            st.success("✅ Mix generated successfully!")
                            
                            total_duration = sum(t['duration'] for t in tracks_metadata)
                            st.write(f"📊 **Mix Stats:** {len(tracks_metadata)} tracks | {total_duration/60:.1f} min total source | Target: {average_bpm:.1f} BPM | Volume: -14 LUFS")
                            
                            if transition_log:
                                with st.expander("🎚️ Transition Effects Used"):
                                    for transition in transition_log:
                                        st.write(f"• {transition}")
                            
                            safe_filename = "".join(c for c in safe_mix_name if c.isalnum() or c in (' ', '-', '_')).strip()
                            if not safe_filename:
                                safe_filename = "Mix"
                            
                            st.download_button(
                                label=f"⬇️ Download {safe_filename}.mp3",
                                data=mp3_buffer,
                                file_name=f"{safe_filename}.mp3",
                                mime="audio/mpeg",
                                type="primary",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to create mix.")
                    else:
                        st.error("Not enough valid tracks to create a mix.")
                
                finally:
                    if tracks_metadata and temp_dir:
                        cleanup_temp_files(tracks_metadata, temp_dir)
                    gc.collect()

