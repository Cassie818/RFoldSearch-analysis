"""Build a manuscript-facing summary from the current shared-universe outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import OUT_DIR, read_tsv


def metrics(path: Path) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in read_tsv(path)}


def number(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def percent(value: str, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def pvalue(value: str) -> str:
    value_ = float(value)
    return f"{value_:.2e}" if value_ < 0.0005 else f"{value_:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "results_summary.md")
    args = parser.parse_args()

    base = metrics(OUT_DIR / "base_inclusive_rmsd_summary.tsv")
    pairing = metrics(OUT_DIR / "pairing_pattern_stratification_summary.tsv")
    bgsu = metrics(OUT_DIR / "bgsu_random_window_background_summary.tsv")

    lines = [
        "# Recurrent 3Dr word controls: shared-universe results",
        "",
        f"The analysis contains {base['n_candidates']} recurrent four-token words. All RMSD distributions use {base['eligible_universe']}.",
        "",
        "## Structural conservation",
        "",
        f"- Five-anchor median RMSD: candidates {number(base['five_anchor_candidate_median_rmsd'])} Å; frequency-matched controls {number(base['five_anchor_frequency_matched_median_rmsd'])} Å; length-matched random windows {number(base['five_anchor_length_random_median_rmsd'])} Å; sequence-matched windows {number(base['five_anchor_sequence_matched_median_rmsd'])} Å.",
        f"- Base-inclusive median RMSD: candidates {number(base['base_inclusive_candidate_median_rmsd'])} Å; frequency-matched controls {number(base['base_inclusive_frequency_matched_median_rmsd'])} Å; length-matched random windows {number(base['base_inclusive_length_random_median_rmsd'])} Å; sequence-matched windows {number(base['base_inclusive_sequence_matched_median_rmsd'])} Å.",
        f"- Candidate versus frequency-matched permutation tests: five-anchor P = {pvalue(base['permutation_five_anchor_candidate_vs_frequency_control_p'])}; base-inclusive P = {pvalue(base['permutation_base_inclusive_candidate_vs_frequency_control_p'])} ({base['permutation_base_inclusive_candidate_vs_frequency_control_n_permutations']} permutations).",
        "",
        "## Pairing categories",
        "",
        f"Biotite annotations assigned all {pairing['n_candidates']} words to one of four modal categories: standard helical {pairing['category_a_n_words']}, fully paired non-standard {pairing['category_b_n_words']}, partially unpaired {pairing['category_c_n_words']} and fully unpaired {pairing['category_d_n_words']}.",
        "",
        "## BGSU overlap",
        "",
        f"Using the at-least-one-residue definition, the observed median per-word overlap was {percent(bgsu['threshold_1_candidate_word_median_fraction'])}, versus {percent(bgsu['threshold_1_null_word_median_fraction'])} under the occurrence-matched random-window null (one-sided empirical P = {pvalue(bgsu['threshold_1_word_median_empirical_p_ge'])}; {bgsu['n_repeats']} replicates).",
        f"Standard-helical words: observed {percent(bgsu['threshold_1_group_a_observed_word_median_fraction'])}, null {percent(bgsu['threshold_1_group_a_null_word_median_fraction'])}, P = {pvalue(bgsu['threshold_1_group_a_empirical_p_ge'])}. All other words: observed {percent(bgsu['threshold_1_group_nonhelix_observed_word_median_fraction'])}, null {percent(bgsu['threshold_1_group_nonhelix_null_word_median_fraction'])}, P = {pvalue(bgsu['threshold_1_group_nonhelix_empirical_p_ge'])}.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
