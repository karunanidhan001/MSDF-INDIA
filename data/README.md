# `data/` — what is here

Small, versionable pieces of the MSDF-India release. The 61 GiB of `.npy` feature arrays
stay on Drive; everything needed to index, split, evaluate and audit the corpus is here.

| Path | Rows / files | Source |
|---|---|---|
| `clip_metadata/<partition>_<language>.csv` | 24 files, 58,597 rows total | the released `metadata.csv` of each leaf folder, verbatim |
| `summary_features/<partition>_<language>.csv` | 24 files, 58,597 rows total | the released `summary_features.csv` of each leaf folder, verbatim — 13 scalars per clip |
| `splits/benchmark_split.csv` | 293 | song-disjoint train/test assignment, seed 42 |
| `splits/balanced_test_clips.csv` | 360 | language-balanced benchmark, 60 bonafide clips per language, seed 7 |
| `inventory/recording_index.csv` | 1,158 | recording-level index of the generation store (294 real + 864 deepfake) with the Drive paths |
| `inventory/drive_files_{rvc,fft,sovits_anime}.csv` | 270 / 283 / 160 | per-file Drive inventory of the generated-audio store, with file IDs |
| `templates/source_provenance_TEMPLATE.csv` | 293 | **to be filled in** — singer, gender, source URL, licence per bonafide recording |

`clip_metadata/` and `summary_features/` are byte-for-byte what the release contains,
including the corrupted `language` column described in
[../docs/DATASET_CARD.md](../docs/DATASET_CARD.md#known-release-defects). Corrections are
applied by the loader, not by editing these files, so the repository stays an honest mirror
of what a user downloads.

Bengali So-VITS is empty in the release; `clip_metadata/sovits_bengali.csv` and
`summary_features/sovits_bengali.csv` are correspondingly empty and are kept so that the
24-cell grid is complete.

## The recording-level index vs the clip-level metadata

`inventory/recording_index.csv` counts **source files** in the generation store (FFT 426,
RVC 278, So-VITS 160, real 294). `clip_metadata/` counts **10 s clips in the feature
release**. They describe different populations and differ slightly even at the file level —
e.g. FFT 426 recordings there against 424 source files with released features. Do not mix
the two in one table without saying which is which.
