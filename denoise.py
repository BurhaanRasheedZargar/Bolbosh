import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from resemble_enhance.enhancer.inference import denoise, enhance

# --- CONFIGURATION ---
INPUT_DIR = Path("data/Kashmiri/wavs") # Change to your actual path
OUTPUT_DIR = Path("data/Kashmiri/wavs_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_dataset():
    wav_files = list(INPUT_DIR.glob("*.wav"))
    print(f"[*] Found {len(wav_files)} files. Processing on {DEVICE}...")

    for wav_path in tqdm(wav_files):
        try:
            # Load audio
            dwav, sr = torchaudio.load(wav_path)
            dwav = dwav.mean(dim=0)

            # Enhance: applies both denoising and dereverberation
            # If it sounds too artificial, switch 'enhance' to 'denoise'
            # Point directly to the stage2 folder where hparams.yaml lives
            model_path = Path(os.path.expanduser("~/.conda/envs/denoise-env/lib/python3.10/site-packages/resemble_enhance/model_repo/enhancer_stage2"))
            
            # Pass run_dir to the function
            hwav, new_sr = enhance(dwav, sr, device=DEVICE, nfe=32, solver="midpoint", lambd=0.5, tau=0.5, run_dir=model_path)

            # Save clean audio
            out_path = OUTPUT_DIR / wav_path.name
            torchaudio.save(out_path, hwav.cpu().unsqueeze(0), new_sr)
            
        except Exception as e:
            print(f"Error processing {wav_path.name}: {e}")

if __name__ == "__main__":
    process_dataset()