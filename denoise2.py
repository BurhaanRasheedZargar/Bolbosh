import librosa
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
 
# Suppress librosa warnings about PySoundFile for clean terminal output
warnings.filterwarnings('ignore')
 
# --- CONFIGURATION ---
INPUT_DIR = Path("data/Kashmiri/wavs_clean")     # From your denoising step
OUTPUT_DIR = Path("data/Kashmiri/wavs_final")    # The final destination
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
TARGET_LUFS = -23.0
TARGET_SR = 22050  # Matcha-TTS LJSpeech config expects 22.05 kHz
 
def process_file(wav_path):
    try:
        # 1. Load audio and force sample rate to 22050Hz (Matcha-TTS standard)
        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
 
        # 2. Trim Silence
        # top_db=40 means anything 40 decibels quieter than the peak volume is considered silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=40, frame_length=1024, hop_length=256)
 
        # Skip if the file is completely empty or just a micro-glitch (less than 0.2 seconds)
        if len(y_trimmed) < sr * 0.2:
            return f"Skipped (Too Short): {wav_path.name}"
 
        # 3. Volume Normalization (LUFS)
        meter = pyln.Meter(sr) 
        loudness = meter.integrated_loudness(y_trimmed)
        # Prevent math errors on pure silence clips that slipped through
        if np.isinf(loudness):
            return f"Skipped (Infinite Loudness): {wav_path.name}"
        y_normalized = pyln.normalize.loudness(y_trimmed, loudness, TARGET_LUFS)
 
        # 4. Save Final Audio
        out_path = OUTPUT_DIR / wav_path.name
        sf.write(out_path, y_normalized, sr)
        return None # Success
    except Exception as e:
        return f"Error on {wav_path.name}: {str(e)}"
 
def main():
    wav_files = list(INPUT_DIR.glob("*.wav"))
    print(f"[*] Found {len(wav_files)} files. Starting Multiprocessing (8 Cores)...")
 
    errors = []
    # Process files across all available CPU cores
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Use tqdm to show a progress bar
        results = list(tqdm(executor.map(process_file, wav_files), total=len(wav_files)))
    # Collect and print any skips/errors
    for r in results:
        if r is not None:
            errors.append(r)
 
    print(f"\n[*] Processing Complete! Saved to {OUTPUT_DIR}")
    if errors:
        print(f"[*] Note: {len(errors)} files were skipped or had errors.")
        # Print first 5 errors as a sample
        for e in errors[:5]:
            print(f"    - {e}")
 
if __name__ == "__main__":
    main()