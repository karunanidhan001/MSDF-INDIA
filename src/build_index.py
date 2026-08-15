"""Build a single clip-level index from the 24 released metadata files.

The MSDF-India feature release stores one `metadata.csv` per language x partition
folder. This module concatenates them into one DataFrame and adds the two columns
that the per-folder files leave implicit: `partition` and `content_class`.

Usage:
    python src/build_index.py                 # print the inventory tables
    python src/build_index.py --out index.csv # also write the merged index
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_METADATA_DIR = os.path.join(REPO_ROOT, "data", "clip_metadata")

LANGUAGES = ["hindi", "punjabi", "bengali", "telugu", "odia", "tamil"]
PARTITIONS = ["real", "rvc", "fft", "sovits"]

# Columns of every released metadata.csv, in order.
RELEASED_COLUMNS = [
    "sample_id",
    "language",
    "label",
    "method",
    "original_filename",
    "clip_index",
    "duration_seconds",
]


def load_index(metadata_dir: str = CLIP_METADATA_DIR) -> pd.DataFrame:
    """Concatenate the 24 released metadata files into one clip index.

    Two corrections are applied, both documented in docs/DATASET_CARD.md:

    1. `language` is repaired from the folder the file came from. In 12 of the 24
       released files the column carries the source audio's parent directory name
       instead of the language -- `vocals`, `references` or, for the So-VITS
       partitions of Odia/Punjabi/Tamil/Telugu, `output` for every row. Grouping on
       the raw column silently drops 6,266 So-VITS clips. The value as released is
       preserved in `language_as_released`.
    2. `partition` is added, since the per-folder files do not name themselves.

    Empty partitions (Bengali So-VITS) contribute no rows but are not an error.
    """
    frames = []
    for partition in PARTITIONS:
        for language in LANGUAGES:
            path = os.path.join(metadata_dir, f"{partition}_{language}.csv")
            if not os.path.exists(path) or os.path.getsize(path) < 10:
                continue
            frame = pd.read_csv(path)
            missing = set(RELEASED_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            frame["partition"] = partition
            frame["language_as_released"] = frame["language"]
            frame["language"] = language
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no metadata files found under {metadata_dir}")

    index = pd.concat(frames, ignore_index=True)
    from filter_synthetic import add_content_class  # local import: same package dir

    return add_content_class(index)


def inventory(index: pd.DataFrame) -> pd.DataFrame:
    """Clips per language x partition, with row and column totals."""
    table = index.pivot_table(
        index="language",
        columns="partition",
        values="sample_id",
        aggfunc="count",
        fill_value=0,
    )
    table = table.reindex(index=LANGUAGES, columns=PARTITIONS, fill_value=0)
    table["total"] = table.sum(axis=1)
    table.loc["TOTAL"] = table.sum(axis=0)
    return table.astype(int)


def source_counts(index: pd.DataFrame) -> pd.DataFrame:
    """Distinct source recordings per language x partition."""
    table = index.pivot_table(
        index="language",
        columns="partition",
        values="original_filename",
        aggfunc="nunique",
        fill_value=0,
    )
    return table.reindex(index=LANGUAGES, columns=PARTITIONS, fill_value=0).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=CLIP_METADATA_DIR)
    parser.add_argument("--out", help="write the merged index to this CSV")
    args = parser.parse_args()

    index = load_index(args.metadata_dir)

    print(f"clips: {len(index):,}   unique sample_id: {index.sample_id.nunique():,}")
    print(f"source recordings: {index.original_filename.nunique():,}")
    print(f"clip duration values: {sorted(index.duration_seconds.unique())}")

    mislabelled = index[index.language != index.language_as_released]
    if len(mislabelled):
        print(
            f"\nrepaired `language` on {len(mislabelled):,} clips "
            f"(as released: {sorted(mislabelled.language_as_released.unique())})"
        )
    print("\nClips per language x partition\n")
    print(inventory(index).to_string())
    print("\nSource recordings per language x partition\n")
    print(source_counts(index).to_string())
    print("\nClips per content class\n")
    print(index.content_class.value_counts().to_string())

    if args.out:
        index.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
