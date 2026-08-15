"""Attach, verify and (optionally) redraw the MSDF-India evaluation splits.

The released protocol is the pair of CSVs under `data/splits/`:

  benchmark_split.csv        song-disjoint train/test assignment, 293 bonafide
                             recordings, drawn with seed 42, ~30% of the
                             recordings of each language held out
  balanced_test_clips.csv    language-balanced benchmark, 60 bonafide clips per
                             language (360 total) drawn with seed 7 from the test
                             recordings only

Those two files ARE the protocol: to compare against published numbers, load them
rather than redrawing. `draw_balanced()` is provided for studies that need a
different balanced draw, and `verify()` checks the shipped files against the
released clip metadata.

Usage:
    python src/make_splits.py                    # verify the shipped protocol
    python src/make_splits.py --redraw 100 --seed 11 --out my_benchmark.csv
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(REPO_ROOT, "data", "splits")
BENCHMARK_SPLIT = os.path.join(SPLITS_DIR, "benchmark_split.csv")
BALANCED_TEST = os.path.join(SPLITS_DIR, "balanced_test_clips.csv")


def normalise_title(text: str) -> str:
    """Fold a song title or filename to a comparable key.

    `benchmark_split.csv` stores cleaned titles while the clip metadata stores raw
    filenames (`004_Hello Brother.mp3.mp3`), so the two are joined on this key.
    """
    text = str(text).lower()
    text = re.sub(r"\.(mp3|wav)", "", text)
    text = re.sub(r"^\d{1,4}_", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def attach_split(index: pd.DataFrame, split: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add a `split` column to the bonafide clips of `index`.

    Synthetic clips receive NaN: their filenames do not all carry a title, so they
    cannot be mapped to a bonafide recording from the release alone (see
    docs/DATASET_CARD.md, "Mapping feature files to source recordings").
    """
    if split is None:
        split = pd.read_csv(BENCHMARK_SPLIT)

    lookup = {
        (row.Language.lower(), normalise_title(row.SongTitle)): row.Split
        for row in split.itertuples()
    }
    index = index.copy()
    keys = [
        (lang, normalise_title(fn)) if partition == "real" else None
        for lang, fn, partition in zip(
            index.language, index.original_filename, index.partition
        )
    ]
    index["split"] = [lookup.get(k) if k is not None else None for k in keys]
    return index


def draw_balanced(
    index: pd.DataFrame, n_per_language: int = 60, seed: int = 7
) -> pd.DataFrame:
    """Draw `n_per_language` bonafide clips per language from the test recordings."""
    if "split" not in index.columns:
        index = attach_split(index)
    pool = index[(index.partition == "real") & (index.split == "test")]
    draws = [
        group.sample(n=min(n_per_language, len(group)), random_state=seed)
        for _, group in pool.groupby("language")
    ]
    return pd.concat(draws, ignore_index=True)


def draw_balanced_spoof(
    index: pd.DataFrame, n_per_cell: int = 20, seed: int = 7
) -> pd.DataFrame:
    """Draw `n_per_cell` conversion-output clips per language x method.

    The release balances only the bonafide side; a real-vs-fake benchmark also
    needs the spoof side balanced, which is the open item recorded in
    docs/EVALUATION_PROTOCOL.md. This draw fills that gap with two caveats that
    must be stated wherever it is used:

    * Cells smaller than `n_per_cell` contribute everything they have (Tamil RVC
      has 18 clips, So-VITS Punjabi 21, Bengali So-VITS none).
    * The draw is NOT guaranteed song-disjoint from the bonafide training
      recordings. Synthetic filenames outside Bengali RVC and the So-VITS branches
      carry no title, so they cannot be traced to a source recording from the
      release alone.
    """
    from filter_synthetic import conversion_outputs_only

    pool = conversion_outputs_only(index)
    pool = pool[pool.partition != "real"]
    draws = [
        group.sample(n=min(n_per_cell, len(group)), random_state=seed)
        for _, group in pool.groupby(["language", "partition"])
    ]
    return pd.concat(draws, ignore_index=True) if draws else pool.head(0)


def verify(index: pd.DataFrame) -> bool:
    """Check the shipped protocol against the released clip metadata."""
    split = pd.read_csv(BENCHMARK_SPLIT)
    balanced = pd.read_csv(BALANCED_TEST)
    indexed = attach_split(index, split)
    real = indexed[indexed.partition == "real"]
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(condition)
        print(f"  [{'ok' if condition else 'FAIL'}] {label}{'  ' + detail if detail else ''}")

    print("Song-disjoint split")
    unmapped = int(real.split.isna().sum())
    check("every bonafide clip maps to a split", unmapped == 0, f"unmapped={unmapped}")
    check("split covers 293 recordings", len(split) == 293, f"rows={len(split)}")
    check(
        "clip counts agree with the metadata",
        int(split.Clips.sum()) == len(real),
        f"{int(split.Clips.sum()):,} vs {len(real):,}",
    )
    overlap = (
        real.groupby("original_filename").split.nunique().gt(1).sum()
    )
    check("no recording spans both sides", overlap == 0, f"recordings={overlap}")

    print("\nPer-language split")
    per_language = (
        real.groupby(["language", "split"]).size().unstack(fill_value=0)
    )
    songs = (
        split.assign(language=split.Language.str.lower())
        .groupby(["language", "Split"])
        .size()
        .unstack(fill_value=0)
    )
    print(per_language.join(songs, rsuffix="_songs").to_string())

    print("\nLanguage-balanced benchmark")
    check("360 clips", len(balanced) == 360, f"rows={len(balanced)}")
    check(
        "60 per language",
        set(balanced.Language.value_counts()) == {60},
        str(dict(balanced.Language.value_counts())),
    )
    known = set(real.sample_id)
    check(
        "all clips exist in the release",
        balanced.sample_id.isin(known).all(),
        f"{balanced.sample_id.isin(known).sum()}/{len(balanced)}",
    )
    test_ids = set(real[real.split == "test"].sample_id)
    from_test = balanced.sample_id.isin(test_ids).sum()
    check(
        "all clips come from test recordings",
        from_test == len(balanced),
        f"{from_test}/{len(balanced)}",
    )
    songs_behind = (
        real[real.sample_id.isin(set(balanced.sample_id))]
        .groupby("language")
        .original_filename.nunique()
    )
    print("\n  recordings behind the 60 balanced clips, per language:")
    print("  " + songs_behind.to_string().replace("\n", "\n  "))
    return ok


def main() -> None:
    from build_index import load_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redraw", type=int, metavar="N", help="draw N clips per language")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", help="write the redrawn manifest here")
    args = parser.parse_args()

    index = load_index()

    if args.redraw:
        drawn = draw_balanced(index, n_per_language=args.redraw, seed=args.seed)
        print(drawn.groupby("language").size().to_string())
        if args.out:
            drawn[["sample_id", "language", "original_filename", "clip_index"]].to_csv(
                args.out, index=False
            )
            print(f"wrote {args.out}")
        return

    print(f"Verifying the released protocol against {len(index):,} indexed clips\n")
    ok = verify(index)
    print("\nPASS" if ok else "\nFAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
