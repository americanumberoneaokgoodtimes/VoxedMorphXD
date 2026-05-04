import gradio as gr
import numpy as np
import os
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import gc
from voxed import (
    load_audio_mapped, process_large_file, get_timestamp, 
    write_wav_chunked, log, check_dependencies
)

def create_spectrogram(audio, sr):
    plt.figure(figsize=(10, 4))
    # Limit visualization to first 30s to save time
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio[:sr*30])), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (First 30s)')
    plt.tight_layout()
    
    temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plot_path = temp_plot.name
    temp_plot.close()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def process_audio_ui(
    input_file, chunk_sec, intensity, metallic, voicing, 
    pitch_shift, vowel, formant_bias, glottal_model, 
    rd, vibrato, jitter, shimmer, whisper_mode, 
    freq_low, freq_high, time_range, threads
):
    if not input_file:
        return None, None, "No input file provided."
    
    log("=== UI Processing Started ===")
    
    try:
        audio_map, sr, temp_wav_path = load_audio_mapped(input_file)
    except Exception as e:
        return None, None, f"Error loading audio: {e}"

    # Time trimming
    working_audio = audio_map
    if time_range and '-' in time_range:
        from voxed import parse_time_to_seconds
        try:
            start_str, end_str = time_range.split('-', 1)
            start_idx = int(parse_time_to_seconds(start_str) * sr)
            end_idx = int(parse_time_to_seconds(end_str) * sr)
            working_audio = audio_map[max(0, start_idx):min(len(audio_map), end_idx)]
            log(f"  [UI] Trimmed to range: {start_str} to {end_str}")
        except:
            pass

    processed_map, temp_raw_path = process_large_file(
        working_audio, sr, chunk_sec=chunk_sec, num_workers=int(threads),
        intensity=intensity, metallic=metallic, voicing=voicing,
        pitch_shift=pitch_shift, vowel=vowel, formant_bias=formant_bias,
        glottal_model=glottal_model, rd=rd, vibrato=vibrato,
        jitter=jitter, shimmer=shimmer, whisper_mode=whisper_mode,
        freq_mask=(freq_low, freq_high)
    )
    
    out_path = f"UI-Voxed-{get_timestamp()}.wav"
    log(f"  [UI] Writing output to {out_path}")
    write_wav_chunked(out_path, sr, processed_map)
    
    # Cleanup for Windows
    processed_map._mmap.close()
    audio_map._mmap.close()
    del processed_map
    del audio_map
    gc.collect()
    
    try:
        os.remove(temp_raw_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
    except Exception as e:
        log(f"  [UI] Minor cleanup warning: {e}")
        
    # Generate Visualization for output
    log("  [UI] Generating preview spectrogram...")
    output_preview, _ = librosa.load(out_path, sr=sr, duration=30)
    spec_path = create_spectrogram(output_preview, sr)
    
    log("=== UI Processing Complete ===")
    return out_path, spec_path, f"Success: {out_path}"

# Custom Theme and CSS for Forensics Look
css = """
.gradio-container { background-color: #0b0f19; color: #00ff41; font-family: 'Courier New', Courier, monospace; }
.gr-button { background-color: #1a2433; border: 1px solid #00ff41; color: #00ff41; }
.gr-slider { color: #00ff41; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🧬 VoxMorph v6 Forensics UI")
    gr.Markdown("Advanced Audio Transmogrification & Whisper Resurrection")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_audio = gr.Audio(label="Input Source", type="filepath")
            with gr.Accordion("Core Engine Parameters", open=True):
                intensity = gr.Slider(1, 100, value=85, label="Vocal Intensity")
                metallic = gr.Slider(0, 100, value=32, label="Metallic Character")
                voicing = gr.Slider(0, 100, value=80, label="Harmonic Voicing")
                pitch_shift = gr.Slider(-12, 12, value=0, step=0.1, label="Pitch Shift (Semitones)")
                formant_bias = gr.Slider(0.7, 1.45, value=1.15, step=0.01, label="Formant Bias")
                vowel = gr.Dropdown(['neutral','a','e','i','o','u'], value='neutral', label="Target Vowel Template")
            
            with gr.Accordion("Glottal Excitation", open=False):
                glottal_model = gr.Dropdown(['simple','klatt','rosenberg','lf'], value='klatt', label="Model")
                rd = gr.Slider(0.5, 2.5, value=0.9, step=0.1, label="Rd (Waveform Shape)")
                vibrato = gr.Slider(0.0, 0.15, value=0.0, label="Vibrato")
                jitter = gr.Slider(0.0, 0.08, value=0.0, label="Jitter")
                shimmer = gr.Slider(0.0, 0.12, value=0.0, label="Shimmer")
            
            with gr.Accordion("Advanced & Masking", open=False):
                whisper_mode = gr.Checkbox(label="Optimize for Faint Whispers")
                time_range = gr.Textbox(label="Time Range (e.g. 10s-20s)", placeholder="Leave blank for full file")
                freq_low = gr.Slider(0, 8000, value=0, step=10, label="Spectrum Mask Low (Hz)")
                freq_high = gr.Slider(0, 22050, value=22050, step=10, label="Spectrum Mask High (Hz)")
                chunk_sec = gr.Number(value=60, label="Chunk Size (sec)")
                threads = gr.Number(value=6, label="Worker Threads")

            btn = gr.Button("🚀 RUN TRANSMOGRIFICATION", variant="primary")
            
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Processed Output")
            output_spec = gr.Image(label="Output Spectrogram")
            status = gr.Textbox(label="System Log")
            
            with gr.Row():
                gr.Button("Preset: Subtle Recovery").click(lambda: [65, 25, 68, -2, 1.15, 'neutral', True], outputs=[intensity, metallic, voicing, pitch_shift, formant_bias, vowel, whisper_mode])
                gr.Button("Preset: Robotic").click(lambda: [78, 68, 82, 2, 1.15, 'neutral', False], outputs=[intensity, metallic, voicing, pitch_shift, formant_bias, vowel, whisper_mode])
                gr.Button("Preset: Extreme").click(lambda: [98, 55, 96, -0.8, 1.25, 'neutral', True], outputs=[intensity, metallic, voicing, pitch_shift, formant_bias, vowel, whisper_mode])

    btn.click(
        process_audio_ui,
        inputs=[
            input_audio, chunk_sec, intensity, metallic, voicing, 
            pitch_shift, vowel, formant_bias, glottal_model, 
            rd, vibrato, jitter, shimmer, whisper_mode, 
            freq_low, freq_high, time_range, threads
        ],
        outputs=[output_audio, output_spec, status]
    )

if __name__ == "__main__":
    check_dependencies()
    demo.launch(share=False)
