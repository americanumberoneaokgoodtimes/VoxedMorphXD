# VoxMorph (Voxed) - Project Context

VoxMorph is a specialized audio processing tool designed to "transmogrify" non-human sounds or faint whispers into human-like speech. It is particularly optimized for resurrecting extremely faint whispering speech buried in high noise floors.

## Project Overview

- **Core Goal**: Extract and exaggerate vocal characteristics from noisy or non-vocal audio sources.
- **Primary Technologies**: 
    - **Language**: Python 3.8+
    - **Libraries**: `numpy`, `scipy`
    - **External Tools**: `ffmpeg` (required for FLAC support and conversion)
- **Key Techniques**:
    - **Memory Mapping & Streaming**: Uses `np.memmap` for both input reading and intermediate output assembly, enabling the processing of multi-GB files on limited-RAM systems.
    - **Formant Hijacking**: Cepstral envelope extraction and LPC (Linear Predictive Coding) pole shifting.
    - **Glottal Pulse Injection**: Synthetic excitation using models like Klatt, Rosenberg, and LF.
    - **Seamless Reassembly**: Refined Overlap-Add (OLA) with linear cross-fading to eliminate windowing artifacts.

## Requirements

- Python 3.8 or higher.
- `numpy`, `scipy`, `tqdm`.
- `ffmpeg` (required for FLAC support).

## Features & UX
- **Progress Bars**: Real-time tracking of chunk processing via `tqdm`.
- **Parallel Processing**: Multi-core utilization via `ProcessPoolExecutor`.
- **Large File Support**: True "multi-GB" handling through disk-based streaming.
- **Sample Rate Aware**: Correctly handles various input sample rates (verified template scaling).
- **Dependency Checks**: Automatic detection of FFmpeg for FLAC compatibility.

### Example Commands
```bash
# Basic whisper recovery
python voxed.py recording.wav --whisper-mode -i 88

# Extreme artistic hallucination
python voxed.py noise.flac -p4 --glottal lf --rd 1.8
```

## Development Conventions

- **Main Script**: `voxed.py` (referenced as `voxmorph_v6.py` in some internal notes).
- **Processing Logic**:
    - Uses STFT/ISTFT for spectral manipulation.
    - Employs Levinson-Durbin recursion for LPC coefficients.
    - Implements multiple glottal excitation models for synthetic voicing.
- **Output Naming**: Files are saved as `<original>-Voxed-<timestamp>.<ext>`.

## TODOs / Future Work
- [ ] Implement a formal test suite (unit tests for LPC/STFT logic).
- [ ] Add support for more audio formats directly via `pydub` or similar if `ffmpeg` dependency needs to be minimized.
- [ ] Optimize glottal pulse generation for speed.
