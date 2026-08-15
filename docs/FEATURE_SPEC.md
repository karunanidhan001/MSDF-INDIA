# Feature specification

All features are computed from the standardised 10 s clips: 16 kHz, mono, peak-normalised,
160,000 samples per clip.

The shapes, dtypes and value ranges below were **measured** from the released arrays. The
extraction settings are those implied by the shapes and ranges; the entries marked
*needs confirmation* cannot be recovered from the arrays alone and should be filled from the
extraction script.

---

## Frame-level arrays (`.npy`, one file per clip per feature)

| Directory | Shape | dtype | Bytes/clip | Settings |
|---|---|---|---:|---|
| `logmel/` | `(80, 1001)` | float32 | 320,448 | 80 Mel bands, `n_fft=400` (25 ms), `hop_length=160` (10 ms), `center=True`; power spectrogram in dB. Observed range `[-59.95, 20.05]`, consistent with `power_to_db(top_db=80)`. |
| `mfcc/` | `(39, 1001)` | float32 | 156,284 | 13 MFCCs with Δ and Δ² (13+13+13), same framing as `logmel`, from the 80-band log-Mel spectrogram. |
| `pitch_frames/` | `(313,)` | float64 | 2,632 | Frame-wise F0 by YIN. `frame_length=2048`, `hop_length=512`, `center=True`. Values bounded exactly by `[50, 1000]` Hz → `fmin=50`, `fmax=1000`. **Unvoiced frames are not flagged** — the estimator returns a value for every frame. |
| `spectral_flux/` | `(312,)` | float32 | 1,376 | Frame-to-frame spectral difference on the same 2048/512 framing; one shorter than the pitch frame count, as a first difference should be. Non-negative. |
| `hnr_frames/` | `(160000,)` | float32 | 640,128 | A per-**sample** array at the clip's full 16 kHz resolution, not per-frame. Non-negative, heavy-tailed (observed max 3.1 × 10⁵) — a harmonic-to-noise *ratio*, not a dB quantity. Its mean and std are released as `hnr_mean` / `hnr_std`. 57 % of the download volume. *Estimator needs confirmation.* |
| `whisper_embeddings/` | `(512,)` | float32 | 2,176 | Whisper encoder hidden states mean-pooled over time. The 512-d width corresponds to the `base` encoder. *Checkpoint and revision need confirmation.* |

**Total ≈ 1.07 MiB per clip → ≈ 61 GiB for all 58,597 clips.**

### Two frame rates, not one

This matters when features are combined:

```
spectrogram family (logmel, mfcc):   n_fft=400,  hop=160   →  1 + 160000/160  = 1001 frames
pitch and flux:                      n_fft=2048, hop=512   →  1 + ⌊160000/512⌋ = 313 frames
                                                              flux = 313 − 1   = 312 frames
hnr:                                 no framing             →  160000 samples
```

### File naming

```
<prefix>_<stem>_clip<k>.npy
```

`<stem>` is `original_filename` with its **final** extension removed — note that bonafide
Bengali/Telugu/Tamil filenames end in `.mp3.mp3`, so one `.mp3` survives into the stem.
`<k>` is `clip_index`, 0-based and unpadded. The same basename is used in all six feature
directories, so one clip is addressed by one name across every feature type.

`<prefix>` is the source audio's parent directory and is **not** a metadata column.
Observed values: `<language>_real`, `<language>_rvc`, `<language>_fft`,
`<language>_sovits_anime` *or* `output_sovits_anime` (varies by language), `references_rvc`,
`references_fft`. Resolve by suffix match — `src/load_features.py` does this for you.

---

## Clip-level summaries (`summary_features.csv`)

13 scalars plus `sample_id`, one row per clip, no missing values. **These ship with this
repository** (`data/summary_features/`), so a large class of analyses needs no download.

| Column | Definition |
|---|---|
| `sample_id` | join key to `metadata.csv` (one-to-one within a folder) |
| `pitch_mean`, `pitch_variance` | mean and variance of `pitch_frames` |
| `pitch_jitter` | mean absolute difference between consecutive pitch frames, `J = 1/(N−1) · Σ |F₀(i+1) − F₀(i)|` |
| `spectral_flux_mean`, `spectral_flux_std` | statistics of `spectral_flux` |
| `hnr_mean`, `hnr_std` | statistics of `hnr_frames` |
| `logmel_mean`, `logmel_std` | grand mean and std over the whole 80 × 1001 array |
| `mfcc_mean`, `mfcc_std` | grand mean and std over the whole 39 × 1001 array |
| `whisper_embedding_norm`, `whisper_embedding_variance` | norm and variance of the 512-d embedding |

---

## Worked example

```
# Real_dataset/tamil/metadata.csv, first row
sample_id,language,label,method,original_filename,clip_index,duration_seconds
f3bbf40fcea681d7d15dbad01f0d6665,tamil,real,real,004_Hello Brother.mp3.mp3,0,10

# the six arrays it resolves to — identical basename in every feature directory
Real_dataset/tamil/logmel/tamil_real_004_Hello Brother.mp3_clip0.npy      (80, 1001) float32
Real_dataset/tamil/mfcc/…                                                (39, 1001) float32
Real_dataset/tamil/pitch_frames/…                                        (313,)     float64
Real_dataset/tamil/spectral_flux/…                                       (312,)     float32
Real_dataset/tamil/hnr_frames/…                                          (160000,)  float32
Real_dataset/tamil/whisper_embeddings/…                                  (512,)     float32

# Real_dataset/tamil/summary_features.csv, matching row
pitch_mean = 136.994   pitch_variance = 29617.367   pitch_jitter = 17.463
spectral_flux_mean = 9.161    spectral_flux_std = 11.775
hnr_mean = 39.481             hnr_std = 1312.259
logmel_mean = -45.702         logmel_std = 18.236
mfcc_mean = -5.177            mfcc_std = 78.724
whisper_embedding_norm = 21.927   whisper_embedding_variance = 0.939
```

Filenames contain spaces, periods and `&` — quote paths.

```python
import sys; sys.path.insert(0, "src")
from build_index import load_index
from load_features import load_feature, with_summary_features

index = load_index()                                  # 58,597 clips
summaries = with_summary_features(index)              # + 13 scalars, no download
array = load_feature(index.iloc[0], "logmel", release_root="/data/MSDF-India")
```

---

## To confirm from the extraction script

1. The HNR estimator and its units.
2. The Whisper checkpoint and revision behind the 512-d embeddings.
3. Library versions used for the released extraction (python, numpy, librosa, torch,
   transformers) — needed for bit-comparable re-extraction.
4. The original sampling rates of the source audio and the resampling method used to reach
   16 kHz.
