# MSDF-India — dataset card

Everything in this document was read from the released Drive tree on **14 August 2026**:
all 24 `metadata.csv` files, all 24 `summary_features.csv` files, and a sample of the
`.npy` arrays. Nothing is carried over unverified from the paper. Re-run
`python src/verify_release.py` to reproduce every number here.

---

## At a glance

| | |
|---|---|
| Languages | Hindi, Punjabi, Bengali, Telugu, Odia, Tamil |
| Generation methods | RVC, FFT spectral manipulation, So-VITS |
| Bonafide source recordings | 294 collected, 293 usable |
| Released clips | **58,597**, every one exactly 10.0 s (162.8 h) |
| Bonafide clips | 18,086 (50.2 h) |
| Clips labelled `fake` | 40,511 (112.5 h) |
| — of which conversion outputs | **32,731** (see [Content classes](#content-classes)) |
| Distribution | feature level only; no waveforms |
| Download size | ≈ 61 GiB (≈ 1.07 MiB per clip) |
| Release | <https://drive.google.com/drive/folders/1bb0pH2RN5EvIYG701BpX3LoiHa3vy1QQ> |

---

## What is in this repository

The repository ships the parts of the release that are small enough to version and that
make the corpus usable without downloading it:

| Path | Contents |
|---|---|
| `data/clip_metadata/` | the 24 released `metadata.csv` files, one per language × partition |
| `data/summary_features/` | the 24 released `summary_features.csv` files — 13 scalars per clip, **all 58,597 clips**, so most analyses need no download at all |
| `data/splits/` | `benchmark_split.csv` and `balanced_test_clips.csv` — the evaluation protocol |
| `data/inventory/` | recording-level index and the Drive file-ID inventories |
| `data/templates/` | the provenance sheet still to be filled in |
| `src/` | loaders, the release audit, split verification, and the scorer |
| `docs/` | this card, the feature spec, the evaluation protocol, the paper supplement |

The frame-level `.npy` arrays are **not** in the repository — they are 61 GiB. Point
`src/load_features.py --release-root` at a local copy of the Drive tree to read them.

---

## Release layout

```
MSDF-India/
├── Real_dataset/
│   └── <language>/                      hindi bengali punjabi tamil telugu odia
│       ├── logmel/  mfcc/  pitch_frames/
│       ├── spectral_flux/  hnr_frames/  whisper_embeddings/
│       ├── metadata.csv
│       └── summary_features.csv
└── Deepfake_dataset/
    ├── RVC/<language>_rvc/              same six feature dirs + 2 CSVs
    ├── FFT/<language>_fft/              same
    └── SoVits_Anime/<language>_sovits_anime/   same
```

24 leaf folders, all with identical internal structure. One language × method cell is one
folder, so partial downloads are straightforward.

---

## Clip inventory

Row counts of the 24 `metadata.csv` files. This is what a user actually gets.

| Language | Bonafide | RVC | FFT | So-VITS | Total |
|---|---:|---:|---:|---:|---:|
| Hindi | 6,634 | 4,188 | 6,626 | 6,634 | 24,082 |
| Punjabi | 2,393 | 148 | 4,037 | 21 | 6,599 |
| Bengali | 2,806 | 2,806 | 2,806 | 0 | 8,418 |
| Telugu | 4,489 | 103 | 4,245 | 4,481 | 13,318 |
| Odia | 1,505 | 33 | 2,100 | 1,505 | 5,143 |
| Tamil | 259 | 23 | 496 | 259 | 1,037 |
| **Total** | **18,086** | **7,301** | **20,310** | **12,900** | **58,597** |

Distinct source files behind those clips:

| Language | Bonafide | RVC | FFT | So-VITS |
|---|---:|---:|---:|---:|
| Hindi | 84 | 64 | 84 | 84 |
| Punjabi | 70 | 71 | 138 | 1 |
| Bengali | 64 | 64 | 64 | 0 |
| Telugu | 50 | 51 | 90 | 50 |
| Odia | 15 | 16 | 28 | 15 |
| Tamil | 10 | 11 | 20 | 10 |

The bonafide side matches the paper exactly: 294 source recordings → 293 usable (one 4.9 s
Punjabi recording falls below the 10 s analysis window) → 18,086 clips at 99.2 % audio
retention.

---

## Content classes

The synthetic folders are **not** homogeneous. Alongside conversion outputs they carry
pipeline by-products that were swept into feature extraction and that also carry
`label=fake`:

| Class | What it is | Clips |
|---|---|---:|
| `bonafide` | the real recordings | 18,086 |
| `converted` | conversion outputs — the actual deepfakes | **32,731** |
| `vocal_stem` | `<title>_vocals.wav`, isolated vocal tracks from the separation stage (FFT branch of Odia/Punjabi/Tamil/Telugu) | 5,662 |
| `reference` | target-voice reference clips (`<lang>_reference.wav`, `<lang>_rvc_ref.wav`) | 24 |
| `unconverted_src` | **Hindi RVC only** — the 32 original recordings sitting beside their `_VOCAL_CONVERTED` counterparts | 2,094 |

Per cell:

| Method | Language | Converted | Vocal stem | Reference | Unconverted |
|---|---|---:|---:|---:|---:|
| RVC | Hindi | 2,094 | – | – | 2,094 |
| RVC | Punjabi | 143 | – | 5 | – |
| RVC | Bengali | 2,806 | – | – | – |
| RVC | Telugu | 98 | – | 5 | – |
| RVC | Odia | 28 | – | 5 | – |
| RVC | Tamil | 18 | – | 5 | – |
| FFT | Hindi | 6,626 | – | – | – |
| FFT | Punjabi | 2,004 | 2,032 | 1 | – |
| FFT | Bengali | 2,806 | – | – | – |
| FFT | Telugu | 2,026 | 2,218 | 1 | – |
| FFT | Odia | 946 | 1,153 | 1 | – |
| FFT | Tamil | 236 | 259 | 1 | – |
| So-VITS | Hindi | 6,634 | – | – | – |
| So-VITS | Telugu | 4,481 | – | – | – |
| So-VITS | Odia | 1,505 | – | – | – |
| So-VITS | Tamil | 259 | – | – | – |
| So-VITS | Punjabi | 21 | – | – | – |
| So-VITS | Bengali | 0 | – | – | – |

**Training or evaluating on the unfiltered synthetic folders inflates the spoof side by
7,780 clips (19 %) and puts 2,094 clips of genuine singing on the spoof side.**
`src/filter_synthetic.py` does the filtering; the benchmark population is
18,086 + 32,731 = **50,817 clips**.

---

## Metadata field definitions

Every leaf folder's `metadata.csv` has these seven fields, one row per clip, no missing
values.

| Field | Definition |
|---|---|
| `sample_id` | 32-character lowercase hex clip identifier. Unique across the whole release (58,597 rows, 58,597 distinct values). **Opaque** — it is not a hash of any released field and cannot be recomputed; treat it as a primary key. Joins to `summary_features.csv` and `balanced_test_clips.csv`. |
| `language` | Language code. **Corrupted in 12 of the 24 files** — see defect 1 below. `src/build_index.py` repairs it from the folder name and keeps the original in `language_as_released`. |
| `label` | `real` or `fake`. `fake` covers every clip in a synthetic folder, including the by-product classes above. |
| `method` | `real`, `rvc`, `fft`, `sovits`. Note the value is `sovits` while the directory is `SoVits_Anime`. |
| `clip_index` | 0-based index of the clip within its source recording; contiguous 0…n−1. |
| `original_filename` | Filename of the audio the clip was cut from, with extension. The **only** pointer back to a source recording. |
| `duration_seconds` | Always 10, across all 58,597 rows. |

`summary_features.csv` adds 13 clip-level scalars keyed by the same `sample_id`; the two
files join one-to-one within every folder. See [FEATURE_SPEC.md](FEATURE_SPEC.md).

---

## Known release defects

All six are detected by `src/verify_release.py`, which fails if any of them silently
changes.

1. **The `language` column is wrong in 12 of the 24 files.** It was derived from the source
   audio's parent directory, so it reads `vocals` or `references` in the FFT and RVC folders
   of Odia/Punjabi/Tamil/Telugu, and `output` for **every** So-VITS row of those four
   languages. 11,952 clips are affected. Grouping on the raw column silently drops 6,266
   So-VITS clips into a phantom "output" language. The folder name is authoritative;
   `build_index.py` repairs it.
2. **7,780 clips labelled `fake` are not deepfakes** — see [Content classes](#content-classes).
   The 2,094 Hindi RVC `unconverted_src` clips are genuine singing on the spoof side.
3. **Bengali So-VITS is empty; Punjabi So-VITS is a single recording (21 clips).** So-VITS
   is a four-language branch in practice, not six.
4. **The So-VITS partition is the anime-preset variant.** The folder is `SoVits_Anime/`,
   filenames are `anime_<preset>_<seq>_<title>`, and the identity token is one of three
   fixed presets assigned round-robin by sequence number — `kawaii_girl`, `soft_anime`,
   `energetic_idol` — not a per-recording target singer. Preset distribution (clips, with
   source recordings in brackets): Hindi 2,894 (28) / 2,267 (28) / 1,473 (28); Telugu 1,366
   (17) / 1,591 (16) / 1,524 (17); Odia 509 (5) / 342 (5) / 654 (5); Tamil 97 (4) / 82 (3) /
   80 (3); Punjabi 21 (1) / – / –. **This must be reconciled with whatever the paper says
   about So-VITS before publication.**
5. **`.npy` paths are not derivable from `metadata.csv`.** File names are
   `<prefix>_<stem>_clip<k>.npy` where `<prefix>` is the source audio's parent directory and
   is not a metadata field. It also varies between languages of the same partition
   (`hindi_sovits_anime` but `output_sovits_anime`, plus `references_rvc` / `references_fft`).
   Join by filename **suffix**; `src/load_features.py` does.
6. **`hnr_frames` is stored per audio sample**, not per frame: 160,000 float32 values per
   clip, 57 % of the whole download, for a signal whose released summary is two scalars.

---

## Mapping clips to source recordings

Because the release is feature level, the link between a synthetic clip and the bonafide
recording behind it lives entirely in `original_filename`. The filename grammars differ by
language and method, and so does the strength of that link.

| Partition | Grammar | Carries the title? |
|---|---|---|
| Bonafide | `<Title>.mp3` (Hindi, Punjabi, Odia) or `<NNN>_<Title>.mp3.mp3` (Bengali, Telugu, Tamil) | — |
| So-VITS, all | `anime_<preset>_<seq>_<Title>.wav` or `anime_<preset>_<seq>_<NNN>_<Title>.mp3.wav` | yes |
| FFT, Hindi | `deepfake_<seq>_<srcid>_<Title>.mp3` | yes, plus a numeric source id |
| FFT / RVC, Bengali | `fft_<seq>_<Title>.mp3_22k.wav`, `rvc_fixed_<seq>_<Title>.mp3_22k.wav` | yes |
| FFT, other languages | `fft_<lang>_<seq>.wav` (outputs) and `<Title>_vocals.wav` (stems) | stems only |
| RVC, other languages | `rvc_<lang>_<seq>.wav` | no |
| RVC, Hindi | `<Voice>.wav` and `<Voice>_VOCAL_CONVERTED.wav` | no — a voice-name namespace |

Resulting coverage — synthetic source files resolvable to a bonafide recording by title:

| Language | RVC | FFT | So-VITS |
|---|---|---|---|
| Hindi | 0/64 (0 %) | 84/84 (100 %) | 84/84 (100 %) |
| Bengali | 64/64 (100 %) | 64/64 (100 %) | — (empty) |
| Punjabi | 0/71 (0 %) | 69/138 (50 %) | 1/1 (100 %) |
| Telugu | 0/51 (0 %) | 45/90 (50 %) | 50/50 (100 %) |
| Odia | 0/16 (0 %) | 14/28 (50 %) | 15/15 (100 %) |
| Tamil | 0/11 (0 %) | 10/20 (50 %) | 10/10 (100 %) |

The 50 % FFT figures are exactly the vocal-stem half. **The fix is a `source_map.csv`
emitted from the generation logs** — `(method, language, original_filename, source_song_id,
source_original_filename)`. It cannot be reconstructed from the released filenames, and
without it the sequence-numbered cells cannot be split song-disjointly against the bonafide
side.

---

## Provenance and singer information — what is missing

| Wanted | Status |
|---|---|
| Performer per recording | **Not in the release.** Bonafide filenames carry the song title, not the singer. |
| Singer gender | **Not in the release.** The repository README reports 142 male / 152 female; no released metadata carries a gender field, so that table cannot currently be reproduced from the data and needs its source attached. |
| Source archive / platform / URL / access date | **Not in the release.** |
| Licence per recording | **Not in the release.** |
| Original sampling rate before the 16 kHz resample | **Not in the release.** |
| Target voice or model per conversion | Only for Hindi RVC, and only as a name in the filename — 32 named voices, ambiguous between source performer and conversion target. |

`data/templates/source_provenance_TEMPLATE.csv` has one row per bonafide recording with
`song_id`, `language`, `song_title`, `original_filename`, `clips` and `split` already filled
from the release, and the eight fields above left empty. Fill it from the collection log —
these are facts about real people and real sources, and are not inferable from a song title.
Regenerate with `python src/make_provenance_template.py`.

---

## Licence and intended use

The corpus is for research on singing-voice deepfake detection and audio forensics. Only
non-invertible derived features are distributed publicly; original audio is available on
request under a controlled-use agreement. Synthetic samples are labelled and must not be
redistributed as authentic performances.
