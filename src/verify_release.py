"""Audit the release against the numbers published in docs/.

Every count in DATASET_CARD.md and in the paper's supplementary material is
asserted here, so that a change to the data shows up as a failing check rather
than as a silent disagreement with the documentation.

Run it after any re-extraction, re-upload or metadata fix:

    python src/verify_release.py
"""
from __future__ import annotations

import sys

import pandas as pd

from build_index import load_index, inventory
from filter_synthetic import conversion_outputs_only
from load_features import load_summary_features
from make_splits import verify as verify_splits

# Expected values, read from the release on 2026-08-14 and published in docs/.
EXPECTED_CLIPS = 58_597
EXPECTED_BONAFIDE = 18_086
EXPECTED_SPOOF_LABELLED = 40_511
EXPECTED_CONTENT_CLASSES = {
    "bonafide": 18_086,
    "converted": 32_731,
    "vocal_stem": 5_662,
    "reference": 24,
    "unconverted_src": 2_094,
}
EXPECTED_INVENTORY = {
    ("hindi", "real"): 6_634, ("hindi", "rvc"): 4_188, ("hindi", "fft"): 6_626, ("hindi", "sovits"): 6_634,
    ("punjabi", "real"): 2_393, ("punjabi", "rvc"): 148, ("punjabi", "fft"): 4_037, ("punjabi", "sovits"): 21,
    ("bengali", "real"): 2_806, ("bengali", "rvc"): 2_806, ("bengali", "fft"): 2_806, ("bengali", "sovits"): 0,
    ("telugu", "real"): 4_489, ("telugu", "rvc"): 103, ("telugu", "fft"): 4_245, ("telugu", "sovits"): 4_481,
    ("odia", "real"): 1_505, ("odia", "rvc"): 33, ("odia", "fft"): 2_100, ("odia", "sovits"): 1_505,
    ("tamil", "real"): 259, ("tamil", "rvc"): 23, ("tamil", "fft"): 496, ("tamil", "sovits"): 259,
}

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {label}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(label)


def main() -> None:
    index = load_index()

    print("Clip index")
    check("clip count", len(index) == EXPECTED_CLIPS, f"{len(index):,}")
    check("sample_id unique", index.sample_id.nunique() == len(index))
    check("every clip is 10 s", set(index.duration_seconds.unique()) == {10})
    check("no missing values", not index[["sample_id", "label", "method", "original_filename", "clip_index"]].isna().any().any())
    check(
        "label vocabulary",
        set(index.label.unique()) == {"real", "fake"},
        str(sorted(index.label.unique())),
    )
    check(
        "method vocabulary",
        set(index.method.unique()) == {"real", "rvc", "fft", "sovits"},
        str(sorted(index.method.unique())),
    )
    check("bonafide clips", int((index.label == "real").sum()) == EXPECTED_BONAFIDE)
    check("clips labelled fake", int((index.label == "fake").sum()) == EXPECTED_SPOOF_LABELLED)

    print("\nPer language x partition")
    table = inventory(index)
    for (language, partition), expected in sorted(EXPECTED_INVENTORY.items()):
        got = int(table.loc[language, partition])
        check(f"{language:8} {partition:7}", got == expected, f"{got:,}")

    print("\nContent classes")
    counts = index.content_class.value_counts().to_dict()
    for name, expected in EXPECTED_CONTENT_CLASSES.items():
        got = int(counts.get(name, 0))
        check(f"{name:16}", got == expected, f"{got:,}")
    clean = conversion_outputs_only(index)
    check(
        "benchmark population",
        len(clean) == EXPECTED_BONAFIDE + EXPECTED_CONTENT_CLASSES["converted"],
        f"{len(clean):,}",
    )

    print("\nSummary features")
    summary = load_summary_features()
    check("one row per clip", len(summary) == len(index), f"{len(summary):,}")
    check("sample_id sets match", set(summary.sample_id) == set(index.sample_id))
    check("13 feature columns", len(summary.columns) - 1 == 13, str(len(summary.columns) - 1))
    check("no NaNs", not summary.isna().any().any())

    print("\nKnown release defects (documented, expected to be present)")
    check(
        "language column corrupted in 12 files",
        int((index.language != index.language_as_released).sum()) == 11_952,
        f"{int((index.language != index.language_as_released).sum()):,} clips",
    )
    check("Bengali So-VITS empty", int(table.loc["bengali", "sovits"]) == 0)
    check("Punjabi So-VITS is one recording", index[(index.language == "punjabi") & (index.partition == "sovits")].original_filename.nunique() == 1)

    print("\nEvaluation protocol")
    verify_splits(index)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for name in failures:
            print(f"  - {name}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
