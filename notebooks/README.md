# Generation notebooks

The six voice-conversion notebooks behind the synthetic partitions, downloaded from the
published Colab links on **15 August 2026** and committed here unmodified — the `.ipynb`
files are byte-for-byte what those Colab links serve, outputs included. Documentation lives
in this file rather than inside the notebooks, so the committed code stays a faithful record
of what was run.

| Notebook | Method | Languages | Colab |
|---|---|---|---|
| [`fft_odia_punjabi_tamil_telugu.ipynb`](fft_odia_punjabi_tamil_telugu.ipynb) | FFT spectral blend | Odia, Punjabi, Tamil, Telugu | [link](https://colab.research.google.com/drive/1TuX07kyC_8T2FK6llZhA39Z-c4UFXPpN) |
| [`fft_hindi_bengali.ipynb`](fft_hindi_bengali.ipynb) | FFT spectral blend | Hindi, Bengali | [link](https://colab.research.google.com/drive/12Q2aWmMRQrCas2JoUc92rhd8qztbQQAS) |
| [`rvc_odia_punjabi_tamil_telugu.ipynb`](rvc_odia_punjabi_tamil_telugu.ipynb) | WORLD vocoder resynthesis | Odia, Punjabi, Tamil, Telugu | [link](https://colab.research.google.com/drive/1O6DneBpPbfXAxxSYtYS0wv-4qmGo9blc) |
| [`rvc_hindi_bengali.ipynb`](rvc_hindi_bengali.ipynb) | WORLD vocoder resynthesis | Hindi, Bengali | [link](https://colab.research.google.com/drive/1XEha1V1gn_hGXsV7kOGavs_NcUgo0HwV) |
| [`sovits_anime_odia_punjabi_tamil_telugu.ipynb`](sovits_anime_odia_punjabi_tamil_telugu.ipynb) | Preset pitch/EQ effect | Odia, Punjabi, Tamil, Telugu | [link](https://colab.research.google.com/drive/126JtKk0LJcP6fjXsnj3woH_Df7nlCquL) |
| [`sovits_anime_hindi_bengali.ipynb`](sovits_anime_hindi_bengali.ipynb) | Preset pitch/EQ effect | Hindi, Bengali | [link](https://colab.research.google.com/drive/1tYg2hWjPN7llQ_DF5_ziCPaHtJHFSpIX) |

File names use the method token of the corresponding release folder (`FFT/`, `RVC/`,
`SoVits_Anime/`) so a notebook can be matched to the data it produced. **The "Method"
column above states what the code actually implements, which is not in every case what the
method name suggest** — see [What each pipeline actually does](#what-each-pipeline-actually-does).

---

## Shared pipeline

All six notebooks follow the same shape and run on Colab against a Drive-mounted corpus:

1. **Install** — `librosa`, `soundfile`, `scipy`, `pyworld`, `tqdm`; the FFT notebooks add
   `demucs`.
2. **Configure** — a `LANGUAGES` dict of input and output directories, one entry per
   language.
3. **Load** the source recordings for each language.
4. **Reference sample** — a 15 s excerpt is taken as the conversion target. In the FFT and
   RVC notebooks this comes from the **first recording of each language**, and conversion
   then runs over the remaining recordings — which is why each FFT/RVC language yields one
   fewer output than it has source recordings.
5. **Convert** every remaining recording against that reference.
6. **Write** the outputs, trimmed (`top_db=20`), peak-normalised, 16-bit PCM at 44.1 kHz.

The FFT notebooks additionally run **Demucs (`htdemucs`)** first and convert the separated
vocal track rather than the full mix.

## What each pipeline actually does

Read from the notebook source, not from the labels:

**FFT** — `librosa.stft` on the separated vocal, a magnitude blend toward the reference
spectrum with the source phase retained, then `librosa.istft`. Classical spectral
manipulation.

**RVC** — `pyworld` analysis/synthesis: `dio` → `stonemask` (F0), `cheaptrick` (spectral
envelope), `d4c` (aperiodicity), envelope and F0 moved toward the reference, then
`pw.synthesize`. This is **WORLD-vocoder resynthesis**. There is no retrieval step, no
content encoder, no feature index and no trained model checkpoint anywhere in either RVC
notebook — nothing that makes a conversion *retrieval-based*. The name matches the release
folder, not the algorithm.

**So-VITS** — a class named `AnimeVoiceConverter` applying `librosa.effects.pitch_shift`
plus brightness/breath EQ and a `tanh` stage, under three fixed presets (`kawaii_girl`,
`soft_anime`, `energetic_idol`) rotated round-robin by index. No SoftVC, no VITS, no
ContentVec, no NSF-HiFi-GAN, no RMVPE — verified absent from **both** So-VITS notebooks,
covering all six languages. This is a DSP effect, and it is what the released
`SoVits_Anime/` features were extracted from.

**These two mismatches are the notebooks' own evidence for the release notes in
[`../docs/DATASET_CARD.md`](../docs/DATASET_CARD.md#known-release-defects), and must be
reconciled with the method descriptions in the paper before publication.**

## Output naming, and which released files each notebook produced

The notebooks' output names are the direct provenance of the released filenames:

| Notebook | Writes | Released cells with that naming |
|---|---|---|
| `fft_odia_punjabi_tamil_telugu` | `fft_<lang>_<NNN>.wav`, `<title>_vocals.wav`, `references/<lang>_reference.wav` | ✅ FFT Odia, Punjabi, Tamil, Telugu |
| `rvc_odia_punjabi_tamil_telugu` | `rvc_<lang>_<NNN>.wav` | ✅ RVC Odia, Punjabi, Tamil, Telugu |
| `rvc_hindi_bengali` | `rvc_<lang>_<NNN>.wav`, `references/<lang>_rvc_ref.wav` | ✅ the `references/` clips; ❌ no released Hindi or Bengali RVC file uses `rvc_<lang>_<NNN>.wav` |
| `sovits_anime_odia_punjabi_tamil_telugu` | `anime_<preset>_<NNN>_<title>.wav` | ✅ So-VITS Odia, Punjabi, Tamil, Telugu |
| `sovits_anime_hindi_bengali` | `anime_<preset>_<NNN>_<title>.wav` | ✅ So-VITS Hindi (Bengali So-VITS is empty in the release) |
| `fft_hindi_bengali` | `fft_<lang>_<NNN>.wav`, `references/<lang>_reference.wav` | ❌ no released Hindi or Bengali FFT file uses this naming |

The released Hindi and Bengali FFT/RVC files carry different names entirely —
`deepfake_<seq>_<srcid>_<Title>.mp3` (Hindi FFT), `fft_<seq>_<Title>.mp3_22k.wav` (Bengali
FFT), `<Voice>_VOCAL_CONVERTED.wav` (Hindi RVC), `rvc_fixed_<seq>_<Title>.mp3_22k.wav`
(Bengali RVC). **So the two Hindi+Bengali FFT/RVC notebooks published here are a separate
run from the one that produced the released Hindi and Bengali features**, and the notebooks
for those releases are not among the six. Worth resolving before the notebooks are cited as
the generation record.

## Two things these notebooks settle

1. **`<title>_vocals.wav` files are Demucs-separated source vocals**, written by the FFT
   notebooks' separation step — not conversion outputs. The 5,662 clips extracted from them
   carry `label=fake` in the release even though they are bonafide-derived.
2. **The `references/` clips are 15 s conversion targets** cut from the first recording of
   each language. The release carries 24 such clips, also labelled `fake`.

Both are filtered by `src/filter_synthetic.py`.

## Notes for anyone re-running these

- Paths are hard-coded to `/content/drive/MyDrive/...`; edit the `LANGUAGES` dict first.
- `fft_odia_punjabi_tamil_telugu.ipynb` ends with two cells unrelated to generation: a
  `pyworld` install and a `getpass` cell that clones this GitHub repository. The token is
  read with `getpass` and its echo is masked, so nothing is leaked — but the cell is
  housekeeping, not part of the pipeline.
- Outputs are 44.1 kHz; the feature extraction resamples everything to 16 kHz.
- Only `rvc_odia_punjabi_tamil_telugu.ipynb` and `fft_odia_punjabi_tamil_telugu.ipynb` ship
  with substantial saved outputs; the rest were cleared before sharing.
