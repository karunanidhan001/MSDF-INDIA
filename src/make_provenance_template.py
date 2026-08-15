"""Generate the per-recording provenance sheet that the release still lacks.

Three things about the bonafide recordings exist nowhere in the release:

  * the performer, and the performer's gender. The repository README reports 142
    male and 152 female singers, but no released metadata carries a singer or
    gender field, so that table cannot currently be reproduced from the data.
  * where each recording came from (archive, platform, URL, access date).
  * the licence each recording is used under.

This script emits `data/templates/source_provenance_TEMPLATE.csv` with one row per
bonafide recording, pre-filled with everything that IS known from the release --
song id, language, title, filename, clip count, split -- and the remaining fields
left EMPTY for the collectors to complete from their records. Nothing is inferred:
a performer's name and gender are facts about a person and must be recorded, not
guessed from a song title.

Usage:
    python src/make_provenance_template.py
"""
from __future__ import annotations

import os

import pandas as pd

from build_index import load_index
from make_splits import BENCHMARK_SPLIT, normalise_title

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO_ROOT, "data", "templates", "source_provenance_TEMPLATE.csv")

# Filled from the release.
KNOWN_COLUMNS = ["song_id", "language", "song_title", "original_filename", "clips", "split"]
# Left empty for the collectors.
TO_FILL_COLUMNS = [
    "singer_name",
    "singer_gender",
    "source_type",
    "source_platform",
    "source_url",
    "access_date",
    "licence",
    "notes",
]


def build() -> pd.DataFrame:
    index = load_index()
    real = index[index.partition == "real"]
    split = pd.read_csv(BENCHMARK_SPLIT)

    filenames = {
        (language, normalise_title(filename)): filename
        for language, filename in real[["language", "original_filename"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    clip_counts = real.groupby(["language", "original_filename"]).size()

    rows = []
    for row in split.itertuples():
        language = row.Language.lower()
        filename = filenames.get((language, normalise_title(row.SongTitle)), "")
        rows.append(
            {
                "song_id": row.SongID,
                "language": language,
                "song_title": row.SongTitle,
                "original_filename": filename,
                "clips": int(clip_counts.get((language, filename), 0)),
                "split": row.Split,
                **{column: "" for column in TO_FILL_COLUMNS},
            }
        )

    frame = pd.DataFrame(rows, columns=KNOWN_COLUMNS + TO_FILL_COLUMNS)
    return frame.sort_values(["language", "song_id"]).reset_index(drop=True)


def main() -> None:
    frame = build()
    frame.to_csv(OUT_PATH, index=False)
    unresolved = int((frame.original_filename == "").sum())
    print(f"wrote {OUT_PATH}")
    print(f"  {len(frame)} bonafide recordings; {unresolved} without a resolved filename")
    print(f"  filled:    {', '.join(KNOWN_COLUMNS)}")
    print(f"  to fill:   {', '.join(TO_FILL_COLUMNS)}")
    print(f"\n  clips accounted for: {frame.clips.sum():,}")
    print(frame.groupby("language").size().to_string())


if __name__ == "__main__":
    main()
