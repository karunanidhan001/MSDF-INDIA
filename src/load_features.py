"""Load MSDF-India features.

Two levels are available:

  clip-level summaries   `data/summary_features/` ships with this repository. 13
                         scalars per clip, no download required.
  frame-level arrays     the six `.npy` directories of the Drive release. Point
                         `--release-root` at a local copy of the release tree.

Resolving a metadata row to its `.npy` file needs care: the file name is
`<prefix>_<stem>_clip<k>.npy`, where `<prefix>` is the directory the source audio
was read from during extraction and is NOT a column of `metadata.csv`. The prefix
also varies between languages of the same partition (`hindi_sovits_anime` but
`output_sovits_anime`). The join is therefore by filename suffix.

Usage:
    python src/load_features.py                                   # summaries only
    python src/load_features.py --release-root /data/MSDF-India   # + one array
"""
from __future__ import annotations

import argparse
import functools
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_DIR = os.path.join(REPO_ROOT, "data", "summary_features")

FEATURES = [
    "logmel",
    "mfcc",
    "pitch_frames",
    "spectral_flux",
    "hnr_frames",
    "whisper_embeddings",
]

# partition -> path of the leaf folder inside the release tree, given a language
FOLDER_TEMPLATE = {
    "real": "Real_dataset/{language}",
    "rvc": "Deepfake_dataset/RVC/{language}_rvc",
    "fft": "Deepfake_dataset/FFT/{language}_fft",
    "sovits": "Deepfake_dataset/SoVits_Anime/{language}_sovits_anime",
}


def load_summary_features(summary_dir: str = SUMMARY_DIR) -> pd.DataFrame:
    """Concatenate the 24 released `summary_features.csv` files."""
    frames = []
    for name in sorted(os.listdir(summary_dir)):
        path = os.path.join(summary_dir, name)
        if not name.endswith(".csv") or os.path.getsize(path) < 10:
            continue
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def with_summary_features(index: pd.DataFrame, summary_dir: str = SUMMARY_DIR) -> pd.DataFrame:
    """Join the clip index to its summary features on `sample_id` (one-to-one)."""
    return index.merge(load_summary_features(summary_dir), on="sample_id", how="left")


def leaf_folder(row, release_root: str) -> str:
    """Absolute path of the release folder a metadata row lives in."""
    template = FOLDER_TEMPLATE[row.partition]
    return os.path.join(release_root, template.format(language=row.language))


@functools.lru_cache(maxsize=None)
def _listing(directory: str) -> tuple[str, ...]:
    return tuple(os.listdir(directory))


def feature_path(row, feature: str = "logmel", release_root: str = ".") -> str:
    """Resolve one metadata row to its `.npy` path inside a local release copy."""
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}; expected one of {FEATURES}")
    directory = os.path.join(leaf_folder(row, release_root), feature)
    stem = row.original_filename.rsplit(".", 1)[0]
    suffix = f"_{stem}_clip{row.clip_index}.npy"
    for name in _listing(directory):
        if name.endswith(suffix):
            return os.path.join(directory, name)
    raise FileNotFoundError(f"no file ending in {suffix!r} under {directory}")


def load_feature(row, feature: str = "logmel", release_root: str = "."):
    import numpy as np

    return np.load(feature_path(row, feature, release_root))


def main() -> None:
    from build_index import load_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", help="local copy of the Drive release tree")
    parser.add_argument("--feature", default="logmel", choices=FEATURES)
    args = parser.parse_args()

    index = load_index()
    joined = with_summary_features(index)
    missing = int(joined.pitch_mean.isna().sum())
    print(f"clips: {len(joined):,}   without summary features: {missing:,}")
    print("\nMean summary features by partition\n")
    columns = ["pitch_mean", "pitch_jitter", "spectral_flux_mean", "hnr_mean", "whisper_embedding_norm"]
    print(joined.groupby("partition")[columns].mean().round(3).to_string())

    if args.release_root:
        row = index.iloc[0]
        array = load_feature(row, args.feature, args.release_root)
        print(f"\n{args.feature}: shape={array.shape} dtype={array.dtype}")
        print(feature_path(row, args.feature, args.release_root))


if __name__ == "__main__":
    main()
