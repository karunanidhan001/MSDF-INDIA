"""Classify and filter the contents of the synthetic partitions.

The three synthetic trees do not contain conversion outputs alone. Alongside them
they carry pipeline by-products that were swept into feature extraction and that
also carry `label=fake`:

  vocal_stem        `<title>_vocals.wav`     isolated vocal tracks from the
                                             source-separation stage (FFT branch of
                                             Odia / Punjabi / Tamil / Telugu)
  reference         `<lang>_reference.wav`,  target-voice reference clips
                    `<lang>_rvc_ref.wav`
  unconverted_src   `<Voice>.wav`            in Hindi RVC only: the original
                                             recordings that sit beside their
                                             `_VOCAL_CONVERTED` counterparts

Anything else in a synthetic folder is a conversion output; everything in the
bonafide folders is `bonafide`.

Training or evaluating on the unfiltered synthetic folders inflates the spoof side
by 7,780 clips (19%) and puts 2,094 clips of genuine singing on the spoof side.

Usage:
    python src/filter_synthetic.py
"""
from __future__ import annotations

import re

import pandas as pd

# `<something>_vocals.wav`
VOCAL_STEM_RE = re.compile(r"_vocals\.(wav|mp3)$", re.IGNORECASE)
# `telugu_reference.wav`, `odia_rvc_ref.wav`, `tamil_rvc_reference.wav`
REFERENCE_RE = re.compile(r"(^|_)ref(erence)?\.(wav|mp3)$", re.IGNORECASE)

CONTENT_CLASSES = ["bonafide", "converted", "vocal_stem", "reference", "unconverted_src"]


def classify(filename: str, method: str, language: str) -> str:
    """Content class of one row, from its filename, method and language."""
    if method == "real":
        return "bonafide"
    if VOCAL_STEM_RE.search(filename):
        return "vocal_stem"
    if REFERENCE_RE.search(filename):
        return "reference"
    # Hindi RVC is the one folder holding both sides of each pair; only the
    # `_VOCAL_CONVERTED` member is a conversion output.
    if method == "rvc" and language == "hindi" and "_VOCAL_CONVERTED" not in filename:
        return "unconverted_src"
    return "converted"


def add_content_class(index: pd.DataFrame) -> pd.DataFrame:
    """Return `index` with a `content_class` column appended."""
    index = index.copy()
    index["content_class"] = [
        classify(fn, method, language)
        for fn, method, language in zip(
            index.original_filename, index.method, index.language
        )
    ]
    return index


def conversion_outputs_only(index: pd.DataFrame) -> pd.DataFrame:
    """Bonafide clips plus conversion outputs; drop every by-product class.

    This is the population the released benchmark is defined on.
    """
    if "content_class" not in index.columns:
        index = add_content_class(index)
    return index[index.content_class.isin(["bonafide", "converted"])]


def main() -> None:
    from build_index import load_index

    index = load_index()
    clean = conversion_outputs_only(index)

    print("Content classes across the release\n")
    breakdown = (
        index.groupby(["partition", "language", "content_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CONTENT_CLASSES, fill_value=0)
    )
    print(breakdown.to_string())

    print("\nTotals per content class\n")
    print(index.content_class.value_counts().reindex(CONTENT_CLASSES).fillna(0).astype(int).to_string())

    dropped = len(index) - len(clean)
    print(
        f"\nkept {len(clean):,} clips "
        f"({(clean.label == 'real').sum():,} bonafide + "
        f"{(clean.label == 'fake').sum():,} spoof); "
        f"dropped {dropped:,} by-product clips"
    )


if __name__ == "__main__":
    main()
