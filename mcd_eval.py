"""
mcd_eval.py — Mel Cepstral Distortion (MCD) evaluation.

Computes MCD using mel-cepstral coefficients extracted via frequency-warped
cepstral analysis (approximating SPTK-style mcep). Scans the generated audio
directory for .wav files, matches each by filename against the ground-truth
directory, and reports MCD with DTW alignment.
"""

import os
import csv
import math
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
GEN_AUDIO_DIR = "/home/gaash/Burhaan/from2022bite008/Matcha-TTS/eval-combine/generated_audio"
GT_WAV_DIR    = "/home/gaash/Burhaan/from2022bite008/Matcha-TTS/data/rasa/Kashmiri/wavs"

# ─── Parameters ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
N_FFT       = 1024
HOP         = 256
MCEP_ORDER  = 24       # number of mel-cepstral coefficients (excluding C0)
ALPHA       = 0.55     # all-pass warping parameter for mel scale at 22050 Hz
OUTPUT_CSV  = "mcd_results.csv"


def load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load a wav file, resample if needed, and peak-normalise."""
    data, sr = sf.read(path, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        duration = len(data) / sr
        n_samples = int(duration * target_sr)
        indices = np.linspace(0, len(data) - 1, n_samples)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(data) - 1)
        frac = indices - idx_floor
        data = data[idx_floor] * (1 - frac) + data[idx_ceil] * frac
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    return data


# ─── Precomputed warping table + window (built once) ─────────────────────────
_OMEGA_ORIG = None
_OMEGA      = None
_WINDOW     = None

def _init_tables():
    global _OMEGA_ORIG, _OMEGA, _WINDOW
    if _OMEGA_ORIG is None:
        n_freqs = N_FFT // 2 + 1
        _OMEGA = np.linspace(0, np.pi, n_freqs)
        # Uniform warped frequencies mapped back to original via inverse all-pass
        omega_w = np.linspace(0, np.pi, n_freqs)
        _OMEGA_ORIG = omega_w - 2.0 * np.arctan2(
            ALPHA * np.sin(omega_w), 1.0 + ALPHA * np.cos(omega_w)
        )
        _OMEGA_ORIG = np.clip(_OMEGA_ORIG, 0.0, np.pi)
        _WINDOW = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT))


def compute_mcep(audio: np.ndarray) -> np.ndarray:
    """
    Extract mel-cepstral coefficients via frequency-warped cepstral analysis.

    1. STFT → log amplitude spectrum
    2. Interpolate onto mel-warped frequency axis (bilinear all-pass, α=0.55)
    3. IFFT of the warped log spectrum → mel-cepstrum
    4. Keep first MCEP_ORDER coefficients (drop C0)

    Returns shape (T, MCEP_ORDER).
    """
    _init_tables()

    pad_len = N_FFT // 2
    audio = np.pad(audio, (pad_len, pad_len), mode="reflect")

    n_frames = 1 + (len(audio) - N_FFT) // HOP
    strides = (audio.strides[0] * HOP, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, N_FFT), strides=strides
    ).copy()
    frames *= _WINDOW

    # Log amplitude spectrum  (n_frames, n_freqs)
    S = np.fft.rfft(frames, n=N_FFT)
    log_amp = np.log(np.maximum(np.abs(S), 1e-10))

    # Warp each frame's spectrum onto the mel-warped frequency axis
    n_freqs = N_FFT // 2 + 1
    warped = np.empty_like(log_amp)
    for i in range(n_frames):
        warped[i] = np.interp(_OMEGA_ORIG, _OMEGA, log_amp[i])

    # Mirror to full symmetric spectrum and IFFT → real cepstrum
    full = np.concatenate([warped, warped[:, -2:0:-1]], axis=1)  # (n_frames, N_FFT)
    cepstrum = np.fft.ifft(full, axis=1).real

    return cepstrum[:, 1 : MCEP_ORDER + 1]   # drop C0, keep first MCEP_ORDER


def dtw_align_mcd(gt: np.ndarray, syn: np.ndarray) -> float:
    """
    Compute MCD in dB between two mel-cepstral sequences using DTW alignment.

    MCD per frame = (10√2 / ln10) · ‖mc_gt − mc_syn‖₂
    Returns mean MCD over the optimal DTW path.
    """
    scale = (10.0 / math.log(10.0)) * math.sqrt(2.0)   # ≈ 6.1415
    T_gt = gt.shape[0]
    T_syn = syn.shape[0]

    # Vectorised pairwise Euclidean distances scaled to MCD
    gt_sq  = np.sum(gt ** 2, axis=1)
    syn_sq = np.sum(syn ** 2, axis=1)
    dist2  = gt_sq[:, None] + syn_sq[None, :] - 2.0 * (gt @ syn.T)
    np.maximum(dist2, 0.0, out=dist2)
    cost = (scale * np.sqrt(dist2)).tolist()

    # DTW forward pass + direction pointers (plain Python for speed)
    dirs = [[0] * T_syn for _ in range(T_gt)]

    prev = [0.0] * T_syn
    prev[0] = cost[0][0]
    for j in range(1, T_syn):
        prev[j] = prev[j - 1] + cost[0][j]
        dirs[0][j] = 2          # came from left

    for i in range(1, T_gt):
        row = cost[i]
        cur = [0.0] * T_syn
        cur[0] = prev[0] + row[0]
        dirs[i][0] = 1          # came from above
        for j in range(1, T_syn):
            d = prev[j - 1]     # diagonal
            u = prev[j]         # up
            l = cur[j - 1]      # left
            if d <= u and d <= l:
                best = d; dirs[i][j] = 0
            elif u <= l:
                best = u; dirs[i][j] = 1
            else:
                best = l; dirs[i][j] = 2
            cur[j] = row[j] + best
        prev = cur

    total_cost = prev[T_syn - 1]

    # Traceback to count path length
    i, j = T_gt - 1, T_syn - 1
    path_len = 1
    while i > 0 or j > 0:
        c = dirs[i][j]
        if c == 0:
            i -= 1; j -= 1
        elif c == 1:
            i -= 1
        else:
            j -= 1
        path_len += 1

    return total_cost / path_len


def main():
    gen_dir = Path(GEN_AUDIO_DIR)
    gt_dir  = Path(GT_WAV_DIR)

    gen_files = sorted(gen_dir.glob("*.wav"))
    if not gen_files:
        print(f"No .wav files found in {gen_dir}")
        return

    print(f"Found {len(gen_files)} generated wav files.")
    print(f"Ground truth directory: {gt_dir}\n")

    gt_names = set(p.name for p in gt_dir.glob("*.wav"))

    pairs = []
    skipped = 0
    for gp in gen_files:
        if gp.name in gt_names:
            pairs.append(gp)
        else:
            skipped += 1

    print(f"Matched {len(pairs)} pairs, {skipped} generated files have no ground truth.\n")

    results = []

    for gen_path in tqdm(pairs, desc="MCD", unit="utt"):
        fname = gen_path.name
        gt_path = gt_dir / fname
        try:
            gt_audio  = load_wav(str(gt_path), SAMPLE_RATE)
            syn_audio = load_wav(str(gen_path), SAMPLE_RATE)

            gt_mcep  = compute_mcep(gt_audio)
            syn_mcep = compute_mcep(syn_audio)

            mcd = dtw_align_mcd(gt_mcep, syn_mcep)
            results.append((fname, mcd))
        except Exception as exc:
            tqdm.write(f"  [error] {fname}: {exc}")
            skipped += 1

    if not results:
        print("\nNo valid pairs were evaluated. Verify your directories.")
        return

    mcds = np.array([r[1] for r in results])

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "mcd_dB"])
        for fname, mcd in results:
            w.writerow([fname, f"{mcd:.4f}"])
    print(f"\nPer-utterance results written to {OUTPUT_CSV}")

    print("\n" + "=" * 50)
    print("  MCD EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Evaluated : {len(results)}")
    print(f"  Skipped   : {skipped}")
    print(f"  Mean MCD  : {np.mean(mcds):.4f} dB")
    print(f"  Std MCD   : {np.std(mcds):.4f} dB")
    print(f"  Median MCD: {np.median(mcds):.4f} dB")
    print(f"  Min MCD   : {np.min(mcds):.4f} dB")
    print(f"  Max MCD   : {np.max(mcds):.4f} dB")
    print("=" * 50)


if __name__ == "__main__":
    main()
