#!/usr/bin/env python3
"""
VoxMorph v6 - Chunked Whisper Resurrection Engine
Handles hours-long, multi-GB files. Optimized for faint whispers in high noise floors.
"""

import argparse
import os
import datetime
import re
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
import subprocess
import sys
import concurrent.futures
from tqdm import tqdm

def parse_time_to_seconds(t):
    """Parses time strings like '27s', '1m30s', '01:10', or '00:00:27' into float seconds."""
    t = t.strip().lower()
    # Handle colon formats (HH:MM:SS or MM:SS)
    if ':' in t:
        parts = list(map(float, t.split(':')))
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        if len(parts) == 2: return parts[0]*60 + parts[1]
        return parts[0]
    
    # Handle suffix formats (27s, 1m, 1h)
    match = re.match(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s?)?$", t)
    if match:
        h, m, s = match.groups()
        return (float(h or 0) * 3600) + (float(m or 0) * 60) + float(s or 0)
    
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"Invalid time format: {t}")

def log(message):
    """Simple timestamped logger"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def check_dependencies():
    log("Checking system dependencies...")
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        log("  [OK] FFmpeg found.")
        return True
    except Exception:
        log("  [WARN] FFmpeg NOT found. FLAC support will be unavailable.")
        return False

def run_ffmpeg_convert(input_path):
    temp_wav = input_path + ".temp.wav"
    log(f"Converting input to compatible WAV format: {input_path} -> {temp_wav}")
    try:
        # Force 16-bit PCM mono WAV
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", temp_wav]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  [ERROR] FFmpeg conversion failed:\n{result.stderr}")
            sys.exit(1)
        log("  [OK] Conversion successful.")
        return temp_wav
    except Exception as e:
        log(f"  [ERROR] An unexpected error occurred during conversion: {e}")
        sys.exit(1)

# ==================== LPC & Formant Magic ====================
def levinson_durbin(r, order):
    a = np.zeros((order + 1, order + 1))
    e = np.zeros(order + 1)
    a[0, 0] = 1.0
    e[0] = r[0]
    for m in range(1, order + 1):
        k = - (r[m] + np.dot(a[1:m, m-1], r[1:m][::-1])) / e[m-1]
        a[0, m] = 1.0
        a[1:m, m] = a[1:m, m-1] + k * a[m-1:0:-1, m-1]
        a[m, m] = k
        e[m] = e[m-1] * (1 - k**2)
    return a[1:, -1], e[-1]

def get_lpc_poles(audio_frame, order=14):
    pre = np.append(audio_frame[0], audio_frame[1:] - 0.975 * audio_frame[:-1])
    corr = np.correlate(pre, pre, mode='full')
    r = corr[len(corr)//2 : len(corr)//2 + order + 1]
    a, _ = levinson_durbin(r, order)
    poles = np.roots(np.concatenate(([1.0], a)))
    return poles[np.abs(poles) < 0.985]

def shift_poles(poles, alpha=1.0, bw_scale=1.0):
    r = np.abs(poles)
    theta = np.angle(poles)
    return (r ** bw_scale) * np.exp(1j * theta * alpha)

def stft(x, n_fft=1024, hop=256):
    return signal.stft(x, nperseg=n_fft, noverlap=n_fft-hop, window='hann', return_onesided=False)

def istft(Zxx, hop=256):
    _, x = signal.istft(Zxx, nperseg=Zxx.shape[0], noverlap=Zxx.shape[0]-hop, window='hann', input_onesided=False)
    return x.real

def get_cepstral_envelope(mag, lifter=22):
    log_mag = np.log(np.abs(mag) + 1e-8)
    cep = np.fft.ifft(log_mag, axis=0)
    cep[lifter:] = 0
    return np.exp(np.real(np.fft.fft(cep, axis=0)))

def human_vocal_template(sr, n_fft=1024, bias=1.0, vowel='neutral'):
    freqs = np.fft.fftfreq(n_fft, d=1/sr)
    env = np.ones(n_fft, dtype=float) * 0.3
    formants = {'neutral': [620, 1380, 2550], 'a': [720, 1100, 2450], 'e': [520, 1850, 2500],
                'i': [310, 2250, 2800], 'o': [550, 850, 2420], 'u': [430, 1050, 2280]}
    peaks = formants.get(vowel, formants['neutral'])
    for f in peaks:
        env += 19 * np.exp(-((np.abs(freqs) - f * bias)**2) / (140**2))
    return env

def generate_glottal_excitation(sr, length, f0, model='klatt', rd=1.0, vibrato=0.0, jitter=0.0, shimmer=0.0):
    t = np.arange(length) / sr
    excitation = np.zeros(length, dtype=np.float32)
    phase = 0.0
    period = sr / f0
    for i in range(length):
        period_var = period * (1 + jitter * np.random.randn()) if jitter > 0 else period
        phase = np.mod(phase + 1/period_var, 1.0)
        vib = 1 + vibrato * np.sin(2 * np.pi * 5.8 * t[i])
        if model == 'klatt':
            if phase < 0.7:
                excitation[i] = phase**2 * np.exp(-3.8 * phase)
            else:
                ret = (phase - 0.7) / 0.3
                excitation[i] = np.exp(-rd * ret * 9)
        elif model == 'rosenberg':
            if phase < 0.6:
                excitation[i] = 0.5 * (1 - np.cos(np.pi * phase / 0.6))
            else:
                ret = (phase - 0.6) / 0.4
                excitation[i] = 0.5 * (1 + np.cos(np.pi * ret)) * np.exp(-rd * ret * 6)
        elif model == 'lf':
            if phase < 0.5:
                excitation[i] = np.sin(np.pi * phase * 2) * 1.3
            else:
                ret = (phase - 0.5) / 0.5
                excitation[i] = -np.exp(-rd * ret * 14)
        if shimmer > 0:
            excitation[i] *= (1 + shimmer * np.random.randn() * 0.6)
    return excitation

def apply_voice_morph_chunk(audio_chunk, sr, intensity=70, metallic=40, voicing=65,
                           pitch_shift=0, vowel='neutral', formant_bias=1.0,
                           glottal_model='klatt', rd=1.0, vibrato=0.0, jitter=0.0, shimmer=0.0,
                           whisper_mode=False):
    """Full core processing with all magic"""
    audio = audio_chunk.astype(np.float32) / (np.max(np.abs(audio_chunk)) + 1e-8)
    intensity = np.clip(intensity / 100.0, 0.01, 1.0)
    metallic = np.clip(metallic / 100.0, 0.0, 1.0)
    voicing = np.clip(voicing / 100.0, 0.0, 1.0)
    
    n_fft, hop = 1024, 256
    _, _, Zxx = stft(audio, n_fft=n_fft, hop=hop)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    source_env = get_cepstral_envelope(mag)
    target_env = human_vocal_template(sr, n_fft=n_fft, bias=formant_bias, vowel=vowel).reshape(-1, 1)
    
    morph_power = intensity**1.1 if whisper_mode else intensity**1.2
    morphed = (1 - morph_power) * mag + morph_power * source_env * target_env * (7.8 if whisper_mode else 7.0)
    
    # LPC pole refinement (protects faint speech)
    if intensity > 0.35 or whisper_mode:
        for i in range(0, len(audio)-n_fft, hop):
            frame = audio[i:i+n_fft]
            if np.std(frame) < (0.01 if whisper_mode else 0.015): continue
            poles = get_lpc_poles(frame)
            if len(poles) >= 5:
                alpha = formant_bias * (0.9 + 0.4 * intensity)
                morphed[:, i//hop] *= (1.18 if whisper_mode else 1.1)
    
    # Glottal excitation
    if voicing > 0.08:
        base_f0 = 108 * (2 ** (pitch_shift / 12.0))
        glottal = generate_glottal_excitation(sr, len(audio), base_f0, glottal_model, rd, vibrato, jitter, shimmer)
        _, _, Gxx = stft(glottal, n_fft=n_fft, hop=hop)
        gain = 5.0 if whisper_mode else 4.3
        morphed *= (1 + voicing * gain * np.abs(Gxx))
    
    final_mag = (1 - metallic) * morphed + metallic * mag * 1.35
    final_mag = signal.medfilt2d(np.abs(final_mag), kernel_size=(3,1))
    
    Zxx_new = final_mag * np.exp(1j * phase)
    processed = istft(Zxx_new, hop=hop)
    processed = processed / (np.max(np.abs(processed)) + 1e-8)
    processed = np.tanh(processed * (1.65 + intensity * 2.5))
    return (processed * 0.9).astype(np.float32)

def load_audio_mapped(file_path):
    temp_wav_path = None
    if file_path.lower().endswith('.flac'):
        temp_wav_path = run_ffmpeg_convert(file_path)
        file_path = temp_wav_path
    
    log(f"Opening file with memory mapping: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            riff = f.read(4)
            if riff != b'RIFF': raise ValueError("Not a WAV file")
            f.read(4) # Size
            wave_head = f.read(4)
            if wave_head != b'WAVE': raise ValueError("Not a WAVE format")
            
            sr, channels, bit_depth, data_offset, data_size = None, None, None, None, None
            
            # Robust Chunk Parsing
            while True:
                chunk_id = f.read(4)
                if not chunk_id: break
                chunk_size = int.from_bytes(f.read(4), 'little')
                
                if chunk_id == b'fmt ':
                    f.read(2) # AudioFormat
                    channels = int.from_bytes(f.read(2), 'little')
                    sr = int.from_bytes(f.read(4), 'little')
                    f.read(6) # ByteRate + BlockAlign
                    bit_depth = int.from_bytes(f.read(2), 'little')
                    if chunk_size > 16: f.read(chunk_size - 16)
                elif chunk_id == b'data':
                    data_offset = f.tell()
                    data_size = chunk_size
                    f.seek(chunk_size, 1) # Skip data to find other chunks if needed
                else:
                    f.seek(chunk_size, 1)
            
            if sr is None or data_offset is None:
                raise ValueError("Could not find required WAV chunks (fmt or data)")

        log(f"  [INFO] Format: {channels} ch | {sr} Hz | {bit_depth} bit")
        dtype = np.int16 if bit_depth == 16 else np.float32
        if bit_depth == 24:
            log("  [WARN] 24-bit WAV detected. This script is optimized for 16-bit or 32-bit float. Results may be distorted.")
        
        num_samples = data_size // (channels * (bit_depth // 8))
        audio_map = np.memmap(file_path, dtype=dtype, mode='r', offset=data_offset, shape=(num_samples, channels) if channels > 1 else (num_samples,))
        log("  [OK] Memory mapping established.")
        return audio_map, sr, temp_wav_path
    except Exception as e:
        log(f"  [ERROR] Failed to map audio file: {e}")
        raise

def process_large_file(audio_map, sr, chunk_sec=30, num_workers=6, **kwargs):
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(2 * sr)
    hop = chunk_samples - overlap_samples
    num_samples = len(audio_map)
    
    out_path = f"temp_out_{get_timestamp()}.raw"
    log(f"Initializing temporary processing buffers: {out_path}")
    processed_map = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(num_samples,))
    weights_path = out_path + ".weights"
    weights_map = np.memmap(weights_path, dtype=np.float32, mode='w+', shape=(num_samples,))
    
    starts = range(0, num_samples, hop)
    duration_hours = num_samples/sr/3600
    log(f"Starting parallel morphing: {duration_hours:.2f} hours | {len(starts)} chunks | {num_workers} workers")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_info = {}
        for start in starts:
            end = min(start + chunk_samples, num_samples)
            chunk = audio_map[start:end]
            if len(chunk.shape) > 1: chunk = np.mean(chunk, axis=1)
            
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            future = executor.submit(apply_voice_morph_chunk, chunk, sr, **kwargs)
            future_to_info[future] = (start, end)
        
        for future in tqdm(concurrent.futures.as_completed(future_to_info), total=len(future_to_info), desc="Morphing", unit="chunk"):
            start, end = future_to_info[future]
            try:
                chunk_proc = future.result()
                chunk_proc = chunk_proc[:end-start]
                
                window = np.ones(len(chunk_proc), dtype=np.float32)
                fade_len = min(overlap_samples // 2, len(chunk_proc) // 4)
                if start > 0:
                    window[:fade_len] = np.linspace(0, 1, fade_len)
                if end < num_samples:
                    window[-fade_len:] = np.linspace(1, 0, fade_len)
                
                processed_map[start:end] += chunk_proc * window
                weights_map[start:end] += window
            except Exception as e:
                log(f"  [ERROR] Chunk at {start} failed: {e}")

    log("Normalizing overlapping regions...")
    processed_map /= np.maximum(weights_map, 1e-8)
    weights_map._mmap.close()
    os.remove(weights_path)
    log("  [OK] Normalization complete.")
    return processed_map, out_path

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%p").lower().replace('am','a').replace('pm','p')

def main():
    parser = argparse.ArgumentParser(description="VoxMorph v6 - Chunked Whisper Resurrection Engine",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Input WAV or FLAC file")
    parser.add_argument("--chunk", type=int, default=60, help="Chunk length in seconds (default 60, use 30-180)")
    parser.add_argument("--time", help="Process only specific time range (e.g., '27s-37s', '01:10-02:00')")
    
    parser.add_argument("-p1", "--preset1", action="store_true", help="Subtle Whisper Recovery")
    parser.add_argument("-p2", "--preset2", action="store_true", help="Robotic")
    parser.add_argument("-p3", "--preset3", action="store_true", help="Strong Recovery")
    parser.add_argument("-p4", "--preset4", action="store_true", help="Extreme Hallucination")
    
    parser.add_argument("-i", "--intensity", type=int, default=85, help="1-100")
    parser.add_argument("-m", "--metallic", type=int, default=32, help="1-100")
    parser.add_argument("-v", "--voicing", type=int, default=80, help="1-100")
    parser.add_argument("-ps", "--pitchshift", type=float, default=0)
    parser.add_argument("--vowel", choices=['neutral','a','e','i','o','u'], default='neutral')
    parser.add_argument("--formant", type=float, default=1.15, help="0.7-1.45")
    
    parser.add_argument("--glottal", choices=['simple','klatt','rosenberg','lf'], default='klatt')
    parser.add_argument("--rd", type=float, default=0.9, help="0.5-2.5")
    parser.add_argument("--vibrato", type=float, default=0.0, help="0.0-0.15")
    parser.add_argument("--jitter", type=float, default=0.0, help="0.0-0.08")
    parser.add_argument("--shimmer", type=float, default=0.0, help="0.0-0.12")
    
    parser.add_argument("--whisper-mode", action="store_true", help="Optimized for faint whispers in high noise")
    parser.add_argument("-t", "--threads", type=int, default=6, help="Number of worker threads (default 6)")
    
    args = parser.parse_args()
    
    log("=== VoxMorph v6 Initialization ===")
    has_ffmpeg = check_dependencies()
    if args.input.lower().endswith('.flac') and not has_ffmpeg:
        log("  [CRITICAL] FFmpeg is required for FLAC processing. Exiting.")
        sys.exit(1)

    # Presets
    if args.preset1:
        args.intensity = 65; args.metallic = 25; args.voicing = 68; args.pitchshift = -2; args.whisper_mode = True
        log("  [INFO] Applied Preset 1: Subtle Whisper Recovery")
    elif args.preset2:
        args.intensity = 78; args.metallic = 68; args.voicing = 82; args.pitchshift = 2
        log("  [INFO] Applied Preset 2: Robotic")
    elif args.preset3:
        args.intensity = 90; args.metallic = 32; args.voicing = 92; args.pitchshift = 4
        log("  [INFO] Applied Preset 3: Strong Recovery")
    elif args.preset4:
        args.intensity = 98; args.metallic = 55; args.voicing = 96; args.pitchshift = -0.8; args.formant = 1.25; args.whisper_mode = True
        log("  [INFO] Applied Preset 4: Extreme Hallucination")
    
    # Load audio
    try:
        audio_map, sr, temp_wav_path = load_audio_mapped(args.input)
    except Exception as e:
        # load_audio_mapped already logs details
        sys.exit(1)
    
    # Handle time trimming
    if args.time:
        try:
            if '-' not in args.time:
                raise ValueError("Time range must be in 'start-end' format (e.g., 27s-37s)")
            start_str, end_str = args.time.split('-', 1)
            start_sec = parse_time_to_seconds(start_str)
            end_sec = parse_time_to_seconds(end_str)
            
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            
            # Bounds check
            total_samples = len(audio_map)
            start_idx = max(0, min(start_idx, total_samples))
            end_idx = max(start_idx, min(end_idx, total_samples))
            
            log(f"Trimming audio to range: {start_sec:.2f}s to {end_sec:.2f}s ({end_idx - start_idx} samples)")
            audio_map = audio_map[start_idx:end_idx]
            
            if len(audio_map) == 0:
                log("  [ERROR] Resulting audio segment is empty. Check your time parameters.")
                sys.exit(1)
        except Exception as e:
            log(f"  [ERROR] Failed to parse time range '{args.time}': {e}")
            sys.exit(1)
    
    log(f"Processing Settings: WhisperMode={args.whisper_mode} | Intensity={args.intensity} | Metallic={args.metallic} | Voicing={args.voicing}")
    
    processed_map, temp_raw_path = process_large_file(audio_map, sr, chunk_sec=args.chunk, num_workers=args.threads,
                                   intensity=args.intensity, metallic=args.metallic, voicing=args.voicing,
                                   pitch_shift=args.pitchshift, vowel=args.vowel, formant_bias=args.formant,
                                   glottal_model=args.glottal, rd=args.rd, vibrato=args.vibrato,
                                   jitter=args.jitter, shimmer=args.shimmer, whisper_mode=args.whisper_mode)
    
    base, ext = os.path.splitext(args.input)
    out_path = f"{base}-Voxed-{get_timestamp()}.wav"
    
    log(f"Saving final output: {out_path}")
    try:
        # We process the final write in chunks to be memory safe
        wavfile.write(out_path, sr, (processed_map[:] * 32767).astype(np.int16))
        log("  [OK] Final file written successfully.")
    except Exception as e:
        log(f"  [ERROR] Failed to save final output: {e}")
    
    log("Cleaning up temporary resources...")
    try:
        processed_map._mmap.close()
        os.remove(temp_raw_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        log("  [OK] Cleanup complete.")
    except Exception as e:
        log(f"  [WARN] Minor cleanup error: {e}")

    log(f"=== Process Finished: {out_path} ===")

if __name__ == "__main__":
    # Prevent multi-processing issues on Windows
    main()
