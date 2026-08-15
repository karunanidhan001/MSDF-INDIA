"""Score a detector under the MSDF-India reporting rules.

The protocol requires four things of any reported number:

  1. EER per language, never only a pooled score.
  2. Any aggregate is the equal-weight mean over languages, so that the largest
     language cannot dominate.
  3. A bootstrap 95% confidence interval on every per-language value.
  4. An explicit statement of the regime -- zero-shot or fine-tuned.

Input is a CSV of `sample_id,score`, where a higher score means "more likely
spoof". Labels, languages and the split come from the released metadata, so a
submission cannot silently disagree with the protocol about which clip is which.

Usage:
    python src/evaluate.py scores.csv --regime zero-shot
    python src/evaluate.py scores.csv --balanced-only
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

LANGUAGES = ["hindi", "punjabi", "bengali", "telugu", "odia", "tamil"]


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate in percent. `labels`: 1 = spoof, 0 = bonafide."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.min() == labels.max():
        return float("nan")  # one class only: EER is undefined

    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    n_spoof = labels.sum()
    n_bona = len(labels) - n_spoof

    # Sweep the threshold down the sorted scores.
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    far = fp / n_bona           # bonafide accepted as spoof
    frr = 1.0 - tp / n_spoof    # spoof rejected as bonafide

    crossing = np.argmin(np.abs(far - frr))
    return float(100.0 * (far[crossing] + frr[crossing]) / 2.0)


def bootstrap_ci(
    labels: np.ndarray, scores: np.ndarray, n_resamples: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the EER, resampling clips with replacement."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    values = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(labels), len(labels))
        value = eer(labels[pick], scores[pick])
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate(
    scored: pd.DataFrame, n_resamples: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Per-language EER with CIs, plus the equal-weight mean across languages.

    `scored` needs columns `language`, `label` and `score`.
    """
    scored = scored.copy()
    scored["y"] = (scored.label == "fake").astype(int)

    rows = []
    for language in LANGUAGES:
        group = scored[scored.language == language]
        if group.empty:
            continue
        value = eer(group.y.values, group.score.values)
        low, high = bootstrap_ci(group.y.values, group.score.values, n_resamples, seed)
        rows.append(
            {
                "language": language,
                "clips": len(group),
                "bonafide": int((group.y == 0).sum()),
                "spoof": int((group.y == 1).sum()),
                "EER%": round(value, 2),
                "CI95_low": round(low, 2),
                "CI95_high": round(high, 2),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table.loc[len(table)] = {
            "language": "MEAN(equal weight)",
            "clips": table.clips.sum(),
            "bonafide": table.bonafide.sum(),
            "spoof": table.spoof.sum(),
            "EER%": round(table["EER%"].mean(), 2),
            "CI95_low": np.nan,
            "CI95_high": np.nan,
        }
    return table


def per_method(scored: pd.DataFrame) -> pd.DataFrame:
    """EER per generation method, scoring each method against all bonafide clips."""
    scored = scored.copy()
    scored["y"] = (scored.label == "fake").astype(int)
    bonafide = scored[scored.y == 0]
    rows = []
    for method in ["rvc", "fft", "sovits"]:
        spoof = scored[(scored.y == 1) & (scored.partition == method)]
        if spoof.empty:
            continue
        pair = pd.concat([bonafide, spoof])
        rows.append(
            {
                "method": method,
                "spoof clips": len(spoof),
                "EER%": round(eer(pair.y.values, pair.score.values), 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    from build_index import load_index
    from filter_synthetic import conversion_outputs_only
    from make_splits import attach_split, BALANCED_TEST

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", help="CSV with columns sample_id,score")
    parser.add_argument("--regime", default="unspecified", choices=["zero-shot", "fine-tuned", "unspecified"])
    parser.add_argument(
        "--balanced-only",
        action="store_true",
        help="the 360-clip balanced bonafide benchmark plus a balanced spoof draw",
    )
    parser.add_argument("--spoof-per-cell", type=int, default=20, help="spoof clips per language x method under --balanced-only")
    parser.add_argument("--test-only", action="store_true", help="restrict to bonafide test-split recordings")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index = attach_split(conversion_outputs_only(load_index()))
    scores = pd.read_csv(args.scores)
    if not {"sample_id", "score"} <= set(scores.columns):
        raise SystemExit("scores CSV needs columns: sample_id,score")

    scored = index.merge(scores[["sample_id", "score"]], on="sample_id", how="inner")
    if args.balanced_only:
        from make_splits import draw_balanced_spoof

        # The released manifest lists the bonafide side only; balance the spoof
        # side here, with the caveats recorded in make_splits.draw_balanced_spoof.
        bonafide = set(pd.read_csv(BALANCED_TEST).sample_id)
        spoof = set(draw_balanced_spoof(index, args.spoof_per_cell, args.seed).sample_id)
        scored = scored[scored.sample_id.isin(bonafide | spoof)]
        print(
            f"balanced benchmark: 60 bonafide/language + up to {args.spoof_per_cell} "
            "spoof clips per language x method"
        )
        print(
            "note: the spoof draw is not guaranteed song-disjoint from the bonafide "
            "training recordings -- see docs/EVALUATION_PROTOCOL.md\n"
        )
    if args.test_only:
        scored = scored[scored.split.isna() | (scored.split == "test")]

    unmatched = len(scores) - len(scored)
    print(f"scored clips: {len(scored):,}   unmatched sample_ids: {unmatched:,}")
    print(f"regime: {args.regime}\n")
    if scored.empty:
        raise SystemExit("no scored clips left after filtering")

    print(evaluate(scored, args.resamples, args.seed).to_string(index=False))
    methods = per_method(scored)
    if not methods.empty:
        print("\nPer generation method\n")
        print(methods.to_string(index=False))


if __name__ == "__main__":
    main()
