# VoxMorph v6 Refactoring & Fix Checklist

## 1. Algorithmic Optimization (Performance)
- [ ] **Vectorize LPC frames**: Replace the frame-by-frame loop in `apply_voice_morph_chunk` with a vectorized approach using `numpy.lib.stride_tricks.as_strided`. (Low priority, current loop is manageable).
- [X] **Vectorize Glottal Excitation**: Refactor `generate_glottal_excitation` to use NumPy array operations instead of a `for i in range(length)` loop.
- [X] **Optimize Levinson-Durbin**: Replace the manual loop with `scipy.linalg.solve_toeplitz` for faster LPC coefficient calculation.

## 2. Memory & I/O Robustness
- [X] **Chunked Output Writing**: Refactor the final `wavfile.write` call. Instead of `processed_map[:]`, write to the output WAV file in manageable chunks to avoid `MemoryError`.
- [X] **Fix Windows File Locking**: Explicitly close all `mmap` instances and use `gc.collect()` before deletion of temporary files.
- [ ] **FFmpeg Progress**: Implement a pipe to capture FFmpeg output and show a progress bar for the conversion stage.

## 3. Signal Processing & Quality
- [X] **Refine Magnitude Morphing**: Improve the spectral blending logic in `apply_voice_morph_chunk` to ensure energy conservation (De-enveloping implemented).
- [X] **Dynamic Pole Count**: Optimized LPC pole handling.
- [X] **Vocal Template Accuracy**: Verified and refined the formant frequencies for templates.

## 4. UI/UX & Integration
- [X] **Unify Parameter Names**: Fixed the inconsistency between `formant`, `formant_bias`, and `bias`.
- [X] **Enhanced Logging**: Added more detail about settings and time ranges.
- [X] **Preset Verification**: Double-checked presets `-p1` to `-p4`.
- [X] **Gradio Web Interface**: Developed `app.py`.
- [X] **Audio Visualization**: Added interactive spectrograms.
- [X] **Spectral Navigator**: Implemented frequency masking controls.

## 5. Documentation & Verification
- [X] **Unit Tests**: Create a small test script to verify `parse_time_to_seconds` and the LPC pole extraction logic.
- [X] **Update README**: Document the new parameters and features.
