# Evaluation protocol

The protocol is the two CSVs in `data/splits/`. To compare against published numbers, load
them — do not redraw. `python src/make_splits.py` verifies both against the released clip
metadata and currently passes every check.

---

## 1. Song-disjoint train/test split

Partitioning is at the **recording** level, never the clip level, so no recording
contributes clips to both sides and a detector cannot ride on recording-specific cues
instead of synthesis artefacts. Seed 42, roughly 30 % of each language's recordings held
out. `benchmark_split.csv` carries the assignment for all 293 usable bonafide recordings
(`SongID`, `Language`, `SongTitle`, `Clips`, `Split`).

| Language | Train songs | Test songs | Train clips | Test clips |
|---|---:|---:|---:|---:|
| Hindi | 59 | 25 | 4,570 | 2,064 |
| Punjabi | 49 | 21 | 1,762 | 631 |
| Bengali | 45 | 19 | 2,030 | 776 |
| Telugu | 35 | 15 | 2,131 | 2,358 |
| Odia | 11 | 4 | 1,046 | 459 |
| Tamil | 7 | 3 | 184 | 75 |
| **Total** | **206** | **87** | **11,723** | **6,363** |

Telugu's test side holds more clips than its train side: the split balances *recordings*,
and Telugu's held-out recordings happen to be long ones.

`benchmark_split.csv` stores cleaned titles while the clip metadata stores raw filenames, so
the two are joined on a normalised key — `src/make_splits.py:normalise_title`, which
resolves all 293.

## 2. Language-balanced benchmark

`balanced_test_clips.csv` — 60 bonafide clips per language, 360 in total, drawn with seed 7
from the **test** recordings only. Verified: all 360 exist in the release and all 360 come
from test-split recordings. Sixty is bounded by Tamil, whose test side offers 75 clips.

Recordings behind each language's 60 clips: Punjabi 19, Hindi 18, Bengali 17, Telugu 10,
Odia 4, **Tamil 3**. Tamil's benchmark is limited in *diversity*, not size — report that
alongside any Tamil result.

## 3. Reporting rules

1. **Per-language EER always** — never only a pooled score.
2. **Aggregate as the equal-weight mean** over languages. A clip-weighted pool re-introduces
   the Hindi bias the balanced draw exists to remove.
3. **Bootstrap 95 % CIs** on every per-language value, ≥ 1,000 resamples.
4. **State the regime** — zero-shot or fine-tuned — for every reported system.

`src/evaluate.py` enforces all four. It takes a `sample_id,score` CSV (higher = more likely
spoof) and takes labels, languages and splits from the released metadata, so a submission
cannot silently disagree about which clip is which.

```bash
python src/evaluate.py scores.csv --regime zero-shot --balanced-only
python src/evaluate.py scores.csv --regime fine-tuned --test-only
```

## 4. Open item — balancing the spoof side

The released manifest balances the **bonafide** side only. A real-vs-fake benchmark also
needs equal spoof clips per language × method, and that cannot be drawn cleanly from the
release as it stands:

* Synthetic filenames outside Bengali RVC and the So-VITS branches carry no title, so
  synthetic clips cannot be traced to a source recording — and therefore cannot be held
  song-disjoint against the bonafide training recordings. See the mapping-coverage table in
  [DATASET_CARD.md](DATASET_CARD.md#mapping-clips-to-source-recordings).
* Several cells are too small to balance against: Tamil RVC has 18 conversion outputs,
  So-VITS Punjabi 21, and Bengali So-VITS none.

`make_splits.draw_balanced_spoof(index, n_per_cell, seed)` draws what is available —
`evaluate.py --balanced-only` uses it and prints the caveat — but this is a stopgap. The
real fix is a `source_map.csv` emitted from the generation logs, after which the spoof side
can be split song-disjointly and balanced properly.

## 5. Published baseline numbers

The paper reports three off-the-shelf SVDD systems evaluated without fine-tuning:

| System | Hindi | Bengali | Punjabi | Tamil | Telugu | Odia | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whisper encoder + ResNet34 | 16.85 | 48.54 | 18.56 | 45.52 | 46.96 | 39.63 | 36.01 |
| Speech Foundation Model Ensemble (E1) | 48.25 | 52.25 | 49.85 | 39.85 | 46.25 | 38.75 | 45.87 |
| XWSB blend | 42.65 | 54.65 | 39.85 | 38.25 | 37.25 | 42.52 | 42.53 |

**These predate the released split.** They were not produced under `benchmark_split.csv` or
the 360-clip balanced draw, and are therefore *not* directly comparable with numbers coming
out of `src/evaluate.py`. Re-running them under the released protocol is an open task; until
that is done, cite them as historical reference points and say so.
