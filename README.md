# MSDF-INDIA

## Multilingual Singing Voice Dataset for Deepfake Singing Voice Detection

**MSDF-INDIA** is a multilingual singing voice corpus developed for research on **deepfake and synthetic singing voice detection**, with a focus on Indian languages.

The corpus contains **294 authentic singing recordings across six Indian languages**, providing linguistic and singer diversity for synthetic audio detection research.

> **Repository contents.** Alongside the generation notebooks, this repository ships the
> released clip metadata, the clip-level summary features for all 58,597 clips, the fixed
> evaluation splits, and the code to index, filter, verify and score the corpus — see
> [Feature-level release](#feature-level-release) and [Quick start](#quick-start) below.
> Start at [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md).

## Corpus Overview

| Language  |    Male |  Female |   Total |
| --------- | ------: | ------: | ------: |
| Bengali   |      33 |      31 |      64 |
| Hindi     |      42 |      42 |      84 |
| Odia      |       6 |       9 |      15 |
| Punjabi   |      33 |      38 |      71 |
| Tamil     |       8 |       2 |      10 |
| Telugu    |      20 |      30 |      50 |
| **Total** | **142** | **152** | **294** |

The corpus contains **142 male and 152 female singers**, providing gender-wise diversity across the six language groups.

> No singer or gender field exists in the released metadata, so this table cannot currently
> be reproduced from the data. `data/templates/source_provenance_TEMPLATE.csv` carries one
> pre-filled row per recording, ready for the per-song singer, gender, source and licence
> details to be entered from the collection records.

---

# Synthetic Singing Voice Generation

Selected voice-conversion experiments were conducted to generate synthetic/deepfake singing voice samples for subsequent detection experiments.

The repository provides **selected experimental notebooks and representative code** to demonstrate the approaches used in the research. The notebooks are provided primarily for **methodological transparency and reviewer reference** rather than as a complete release of every generation configuration.

---

## 1. FFT-Based Voice Conversion

An FFT-based voice conversion experiment was conducted on selected recordings from four language groups.

| Language  | Songs Processed |
| --------- | --------------: |
| Odia      |              13 |
| Punjabi   |              69 |
| Tamil     |               9 |
| Telugu    |              44 |
| **Total** |         **135** |

**Google Colab:**  
https://colab.research.google.com/drive/1TuX07kyC_8T2FK6llZhA39Z-c4UFXPpN?usp=sharing

---

## 2. RVC-Based Voice Conversion

A separate experiment using **Retrieval-based Voice Conversion (RVC)** was conducted on selected recordings.

| Language  | Songs Processed |
| --------- | --------------: |
| Odia      |              14 |
| Punjabi   |              70 |
| Tamil     |               9 |
| Telugu    |              49 |
| **Total** |         **142** |

**Google Colab:**  
https://colab.research.google.com/drive/1O6DneBpPbfXAxxSYtYS0wv-4qmGo9blc?usp=sharing

---

## 3. Multilingual Voice Conversion Experiment

An additional voice-conversion experiment was performed on selected recordings across four language groups.

| Language  | Songs Processed |
| --------- | --------------: |
| Odia      |              15 |
| Punjabi   |               1 |
| Tamil     |              10 |
| Telugu    |              50 |
| **Total** |          **76** |

**Google Colab:**  
https://colab.research.google.com/drive/126JtKk0LJcP6fjXsnj3woH_Df7nlCquL?usp=sharing

---

## Experimental Summary

| Experiment   | Odia | Punjabi | Tamil | Telugu |   Total |
| ------------ | ---: | ------: | ----: | -----: | ------: |
| FFT-Based    |   13 |      69 |     9 |     44 | **135** |
| RVC-Based    |   14 |      70 |     9 |     49 | **142** |
| Multilingual |   15 |       1 |    10 |     50 |  **76** |

**Note:** The numbers in the experimental tables represent **separate processing runs on selected recordings** and should not be added to the 294-song corpus total or interpreted as additional unique songs in the corpus.

---

## Repository Purpose

This repository is provided to support **research transparency and reviewer verification** of the synthetic singing voice generation experiments associated with MSDF-INDIA.

The linked Colab notebooks provide reviewers with access to representative experimental code and the corresponding processing outputs.

The complete set of generation configurations and language-specific implementations is not released as a single turnkey generation package.

## Research Applications

The MSDF-INDIA corpus and associated synthetic samples are intended to support research in:

- Deepfake singing voice detection
- Synthetic audio detection
- Multilingual audio forensics
- Voice-conversion artifact analysis
- Cross-language generalization
- Robustness evaluation of deepfake detection systems

**For research and academic use.**

---

## Dataset Metadata

The complete dataset metadata, including recording-level information and associated dataset attributes, is available here:

[Download / View Dataset Metadata](https://drive.google.com/file/d/1pDCAETf01VipDCS7esICPlul1tnp_Qer/view?usp=sharing)

A copy is versioned here as `data/inventory/recording_index.csv`.

---

# Feature-Level Release

The corpus is distributed at the **feature level** — derived, non-invertible acoustic
features rather than waveforms, so the resource stays reproducible without redistributing
the underlying recordings.

**Release:** <https://drive.google.com/drive/folders/1bb0pH2RN5EvIYG701BpX3LoiHa3vy1QQ>

| | |
|---|---|
| Released clips | **58,597**, each exactly 10.0 s (162.8 h) |
| Bonafide | 18,086 clips from 293 usable recordings |
| Labelled `fake` | 40,511 clips, of which **32,731 are conversion outputs** |
| Features per clip | log-Mel, MFCC+Δ+Δ², F0 frames, spectral flux, HNR, Whisper embedding |
| Download size | ≈ 61 GiB |

Clips per language and partition:

| Language | Bonafide | RVC | FFT | So-VITS | Total |
|---|---:|---:|---:|---:|---:|
| Hindi | 6,634 | 4,188 | 6,626 | 6,634 | 24,082 |
| Punjabi | 2,393 | 148 | 4,037 | 21 | 6,599 |
| Bengali | 2,806 | 2,806 | 2,806 | 0 | 8,418 |
| Telugu | 4,489 | 103 | 4,245 | 4,481 | 13,318 |
| Odia | 1,505 | 33 | 2,100 | 1,505 | 5,143 |
| Tamil | 259 | 23 | 496 | 259 | 1,037 |
| **Total** | **18,086** | **7,301** | **20,310** | **12,900** | **58,597** |

These are clip counts in the feature release, and describe a different population from the
recording-level run counts in the experiment tables above.

### Before you train on it

The synthetic folders also contain pipeline by-products that carry `label=fake`: 5,662
separated vocal stems, 24 target-voice reference clips, and 2,094 clips of **unconverted
Hindi RVC source recordings** — genuine singing on the spoof side. Filtering them out is one
call, and leaves 18,086 bonafide + 32,731 spoof = 50,817 clips:

```python
from build_index import load_index
from filter_synthetic import conversion_outputs_only
clean = conversion_outputs_only(load_index())
```

Five more release-level issues — a corrupted `language` column in 12 of the 24 metadata
files among them — are documented and machine-checked in
[`docs/DATASET_CARD.md`](docs/DATASET_CARD.md#known-release-defects).

---

## Quick start

```bash
pip install -r requirements.txt

python src/build_index.py        # 58,597 clips, inventory tables
python src/filter_synthetic.py   # content classes; drop pipeline by-products
python src/make_splits.py        # verify the released evaluation protocol
python src/verify_release.py     # audit the release against the documented numbers
python src/evaluate.py scores.csv --regime zero-shot --balanced-only
```

Clip-level summary features for **all** 58,597 clips ship in `data/summary_features/`, so
most analyses run without downloading anything:

```python
import sys; sys.path.insert(0, "src")
from build_index import load_index
from load_features import with_summary_features

df = with_summary_features(load_index())   # 58,597 rows, 13 features per clip
```

For the frame-level arrays, point the loader at a local copy of the release:

```python
from load_features import load_feature
array = load_feature(df.iloc[0], "logmel", release_root="/data/MSDF-India")   # (80, 1001)
```

## Repository layout

```
data/
  clip_metadata/        24 released metadata.csv files (58,597 clips)
  summary_features/     24 released summary_features.csv files (13 scalars per clip)
  splits/               benchmark_split.csv, balanced_test_clips.csv
  inventory/            recording-level index + Drive file inventories
  templates/            per-recording provenance sheet to fill in
docs/
  DATASET_CARD.md       inventory, field definitions, known defects, source mapping
  FEATURE_SPEC.md       exact array shapes, framing, naming, worked example
  EVALUATION_PROTOCOL.md  splits, balanced benchmark, reporting rules
  supplementary.tex     supplementary material for the paper
src/
  build_index.py        merge the 24 metadata files into one clip index
  filter_synthetic.py   classify and drop pipeline by-products
  load_features.py      summary features, and .npy resolution by filename suffix
  make_splits.py        verify / redraw the evaluation splits
  evaluate.py           per-language EER with bootstrap CIs
  verify_release.py     assert every documented count against the data
  make_provenance_template.py
```

## Evaluation

Fixed, released, and verifiable — see
[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md):

- **Song-disjoint split** (seed 42, `benchmark_split.csv`): 206 train / 87 test recordings,
  11,723 / 6,363 clips. No recording contributes clips to both sides.
- **Language-balanced benchmark** (seed 7, `balanced_test_clips.csv`): 60 bonafide clips per
  language, 360 total, drawn from test recordings only.
- **Reporting:** per-language EER, equal-weight mean across languages, bootstrap 95 % CIs,
  and an explicit zero-shot / fine-tuned statement.

`python src/make_splits.py` re-verifies all of the above against the released metadata.

## Reproducing the numbers

Every count in the documentation is asserted in code:

```bash
python src/verify_release.py     # exits non-zero if the data and the docs disagree
```

Run it after any re-extraction, re-upload or metadata fix.
