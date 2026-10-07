"""Build a manuscript-facing summary exclusively from generated outputs."""
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
    number_ = float(value)
    return f"{number_:.2e}" if number_ < 0.0005 else f"{number_:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR / "results_summary.md")
    args = parser.parse_args()
    frequency = metrics(OUT_DIR / "frequency_matched_random_words_summary.tsv")
    base = metrics(OUT_DIR / "base_inclusive_rmsd_summary.tsv")
    pairing = metrics(OUT_DIR / "pairing_pattern_stratification_summary.tsv")
    bgsu = metrics(OUT_DIR / "bgsu_random_window_background_summary.tsv")

    candidate_rmsd = float(frequency["candidate_cross_clan_median_rmsd"])
    control_rmsd = float(frequency["cross_clan_matched_control_median_rmsd"])
    below = float(frequency["fraction_candidates_below_cross_clan_control"])
    tighter = candidate_rmsd < control_rmsd and below > 0.5
    sensitivity_tighter = (
        candidate_rmsd < float(frequency["sensitivity_cross_clan_control_median_rmsd"])
        and float(frequency["sensitivity_fraction_candidates_below_control"]) > 0.5
    )
    nonhelix_tighter = (
        float(pairing["nonhelix_candidate_cross_clan_median_rmsd"])
        < float(pairing["nonhelix_control_cross_clan_median_rmsd"])
        and float(pairing["nonhelix_fraction_candidate_below_control"]) > 0.5
    )
    direction = "lower" if tighter else "not lower"
    if tighter and sensitivity_tighter and nonhelix_tighter:
        rmsd_interpretation = (
            "When comparisons are restricted to different-clan occurrence pairs, recurrent words are geometrically tighter than "
            "occurrence-frequency-matched non-candidate words, and the direction holds when control words must "
            "have more cross-clan pairs and when controls are additionally matched by pairing category. The cross-family "
            "diagnostic uses a broader eligible control pool and is therefore not a fixed-pool estimate of the clan restriction."
        )
    elif not tighter and not sensitivity_tighter and not nonhelix_tighter:
        rmsd_interpretation = (
            "Even after comparisons are restricted to different-clan pairs, recurrent words are not tighter than "
            "occurrence-frequency-matched non-candidate words; candidate status alone does not produce lower RMSD."
        )
    else:
        rmsd_interpretation = (
            "The direction of the cross-clan comparison depends on the control definition (primary, minimum-pair "
            "sensitivity and pairing-matched analyses disagree), so no claim about tighter geometry is made."
        )

    lines = [
        "# Recurrent 3Dr word controls: corrected results summary",
        "",
        "Every numerical value below is read from a generated TSV. Primary RMSD comparisons use only pairs from different Rfam clans on both sides; a family without a clan is its own clan.",
        "",
        "## Key results",
        "",
        f"- Cross-clan RMSD: recurrent candidates had median {number(frequency['candidate_cross_clan_median_rmsd'])} Å, versus {number(frequency['cross_clan_matched_control_median_rmsd'])} Å for occurrence-frequency-standardised non-candidate words; {percent(frequency['fraction_candidates_below_cross_clan_control'])} of candidates were below the descriptive matched-control median. A unique-word permutation test stratified by occurrence-frequency bin gave P = {pvalue(frequency['permutation_candidate_vs_cross_clan_control_p'])} ({frequency['permutation_candidate_words']} candidate and {frequency['permutation_control_words']} control words; {frequency['permutation_candidate_vs_cross_clan_control_n_permutations']} permutations).",
        f"- Control homology audit: among the {frequency['n_background_words_occ_ge_8']} non-candidate words occurring at least eight times, {percent(frequency['background_fraction_without_cross_family_pairs'])} had no cross-family pair and {percent(frequency['background_fraction_without_cross_clan_pairs'])} had no cross-clan pair. For the median control word, {percent(frequency['background_same_clan_fraction_of_cross_family_pairs_median'])} of its cross-family pairs fell within one clan. The primary pool retained {frequency['n_background_words_cross_clan_eligible']} words (by frequency bin {frequency['primary_control_pool_by_bin']}); their median number of sampled cross-clan pairs was {number(frequency['primary_control_cross_clan_pairs_median'], 0)}, versus {number(frequency['candidate_cross_clan_pairs_median'], 0)} for candidates.",
        f"- Minimum-pair sensitivity: requiring at least {frequency['sensitivity_min_cross_clan_pairs_per_control_word']} cross-clan pairs per control word (pool {frequency['sensitivity_control_pool_by_bin']}) gave a control median of {number(frequency['sensitivity_cross_clan_control_median_rmsd'])} Å; {percent(frequency['sensitivity_fraction_candidates_below_control'])} of candidates were lower (stratified permutation P = {pvalue(frequency['sensitivity_permutation_p'])}).",
        f"- Cross-family diagnostic (homologous same-clan pairs allowed): {number(frequency['diagnostic_candidate_cross_family_median_rmsd'])} versus {number(frequency['diagnostic_cross_family_matched_control_median_rmsd'])} Å; {percent(frequency['diagnostic_fraction_candidates_below_cross_family_control'])} of candidates were lower.",
        f"- Pairing-matched analysis: after excluding standard helices, {pairing['nonhelix_n_words']} recurrent words had median cross-clan RMSD {number(pairing['nonhelix_candidate_cross_clan_median_rmsd'])} Å, versus {number(pairing['nonhelix_control_cross_clan_median_rmsd'])} Å for controls matched by frequency bin and modal pairing category; {percent(pairing['nonhelix_fraction_candidate_below_control'])} were lower (unique-word stratified permutation P = {pvalue(pairing['nonhelix_permutation_p'])}).",
        f"- BGSU, word-level statistic: under the permissive ≥1-residue rule, the observed median per-word overlap was {percent(bgsu['threshold_1_candidate_word_median_fraction'])}, versus {percent(bgsu['threshold_1_null_word_median_fraction'])} under the word-size-matched null (P10–P90 {percent(bgsu['threshold_1_null_word_median_p10_fraction'])}–{percent(bgsu['threshold_1_null_word_median_p90_fraction'])}; Monte Carlo P = {pvalue(bgsu['threshold_1_word_median_empirical_p_ge'])}).",
        f"- BGSU, pooled diagnostic: the same candidate occurrences gave {percent(bgsu['threshold_1_candidate_pooled_fraction'])}, below the random-window rate of {percent(bgsu['threshold_1_eligible_window_fraction'])}. The word-level and pooled summaries answer different questions and must not be compared as if they were the same statistic.",
        f"- BGSU sensitivity: with ≥3/4 residues required, the pooled candidate fraction was {percent(bgsu['threshold_3_candidate_pooled_fraction'])} versus {percent(bgsu['threshold_3_eligible_window_fraction'])} for random windows; with 4/4 residues it was {percent(bgsu['threshold_4_candidate_pooled_fraction'])} versus {percent(bgsu['threshold_4_eligible_window_fraction'])}. Per-word medians were {percent(bgsu['threshold_3_candidate_word_median_fraction'])} and {percent(bgsu['threshold_4_candidate_word_median_fraction'])} (Monte Carlo P = {pvalue(bgsu['threshold_3_word_median_empirical_p_ge'])} and {pvalue(bgsu['threshold_4_word_median_empirical_p_ge'])}).",
        "",
        "### Pairing-stratum comparison",
        "",
        "| Category | Candidate words | Candidate cross-clan median RMSD (Å) | Category-matched control (Å) | Candidate lower | Holm-adjusted permutation P |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| ({key}) {pairing[f'category_{key}_name']} | {pairing[f'category_{key}_n_words']} | {number(pairing[f'category_{key}_candidate_cross_clan_median_rmsd'])} | {number(pairing[f'category_{key}_control_cross_clan_median_rmsd'])} | {percent(pairing[f'category_{key}_fraction_candidate_below_control'])} | {pvalue(pairing[f'category_{key}_permutation_p_holm'])} |"
            for key in "abcd"
        ],
        "",
        "## Interpretation",
        "",
        rmsd_interpretation,
        "",
        "The BGSU result is statistic-dependent. A typical sparsely observed word is more likely than its size-matched null to have a high permissive overlap fraction, driven by non-helical words, but the pooled occurrence fraction is lower than background and stricter overlap definitions show no enrichment. The defensible conclusion is therefore limited to word-level enrichment under the ≥1-residue rule; broad motif enrichment is not supported.",
        "",
        "## Suggested Results text",
        "",
        f"> Comparing only occurrence pairs from different Rfam clans, recurrent 3Dr words had a median five-anchor RMSD of {number(frequency['candidate_cross_clan_median_rmsd'])} Å, compared with {number(frequency['cross_clan_matched_control_median_rmsd'])} Å for occurrence-frequency-standardised non-candidate words; {percent(frequency['fraction_candidates_below_cross_clan_control'])} of candidates were lower than the descriptive matched-control median. A unique-word permutation test stratified by occurrence-frequency bin gave P = {pvalue(frequency['permutation_candidate_vs_cross_clan_control_p'])}. Among non-helical words with controls additionally stratified by modal pairing category, the medians were {number(pairing['nonhelix_candidate_cross_clan_median_rmsd'])} and {number(pairing['nonhelix_control_cross_clan_median_rmsd'])} Å (stratified permutation P = {pvalue(pairing['nonhelix_permutation_p'])}). Thus, recurrent words were {direction} in RMSD than the matched cross-clan token-word background.",
        "",
        f"> Using the permissive criterion that at least one of four residues intersected a BGSU hairpin or internal-loop annotation, the median overlap fraction across candidate words was {percent(bgsu['threshold_1_candidate_word_median_fraction'])}, compared with {percent(bgsu['threshold_1_null_word_median_fraction'])} in a null distribution that matched each word's number of observable occurrences (Monte Carlo P = {pvalue(bgsu['threshold_1_word_median_empirical_p_ge'])}). In contrast, the pooled candidate occurrence fraction was {percent(bgsu['threshold_1_candidate_pooled_fraction'])}, below the {percent(bgsu['threshold_1_eligible_window_fraction'])} background window rate, and no enrichment was detected when at least three or all four residues were required to overlap. We therefore interpret this as limited word-level enrichment under the permissive overlap definition.",
        "",
        "## Updated Methods text",
        "",
        f"> **Frequency-matched word controls.** We enumerated all non-candidate four-token words occurring at least eight times. Different Rfam families within one clan are treated as related, and {percent(frequency['background_fraction_without_cross_clan_pairs'])} of these words had no occurrence pair from different clans; the primary control pool therefore retained only words with at least {frequency['min_cross_clan_pairs_per_control_word']} cross-clan pair (families without a clan were treated as singleton clans). For descriptive frequency standardisation, {frequency['n_repeats']} controls were sampled with replacement for each candidate from the same occurrence-frequency bin ({frequency['frequency_bins']}) using random seed {frequency['random_seed']}. Candidate and control RMSDs were calculated only between occurrences assigned to different clans, using at most {frequency['max_pairs_per_word']} pairs per word. Statistical inference treated each candidate and control word as one independent unit: word labels were permuted {frequency['permutation_candidate_vs_cross_clan_control_n_permutations']} times within occurrence-frequency bins while preserving the observed candidate and control counts, using the candidate-count-weighted mean of stratum-specific median RMSD differences as the test statistic. Two-sided empirical P values used the plus-one correction. A sensitivity analysis required at least {frequency['sensitivity_min_cross_clan_pairs_per_control_word']} cross-clan pairs per control word. Cross-family, all-pair and token-order permutation results were retained as diagnostics rather than primary tests.",
        "",
        f"> **Pairing-pattern stratification.** Candidate and eligible control words were assigned to standard helical, fully paired non-standard, partially unpaired or fully unpaired categories from their modal Biotite pairing pattern. Descriptive controls were matched jointly by occurrence-frequency bin and modal pairing category. Statistical inference permuted unique word labels within the same frequency-bin-by-category strata using {pairing['n_permutations']} permutations. Category-specific P values were adjusted across the four categories by the Holm method. Both sides used only different-clan occurrence pairs. Pair annotations were available for {pairing['n_occurrences_with_pair_annotations']} candidate occurrences; {pairing['n_occurrences_missing_pair_annotations']} occurrences were excluded for missing annotations.",
        "",
        f"> **BGSU background and sensitivity analysis.** For every candidate word observed on BGSU-covered chains, each null replicate sampled the same number of random four-nucleotide windows as that word's observed denominator. The median overlap fraction across words was then calculated, preserving the candidate word as the inferential unit. Sampling used {bgsu['n_repeats']} replicates, replacement, random seed {bgsu['random_seed']} and {bgsu['n_eligible_random_windows']} eligible windows from {bgsu['n_bgsu_covered_chains']} chains. Overlap thresholds of at least one, at least three and all four residues within the same BGSU motif occurrence were evaluated. Pooled occurrence fractions were reported separately as descriptive sensitivity checks.",
    ]

    if base.get("status") == "complete":
        lines.extend([
            "",
            "## Base-inclusive coordinate diagnostic",
            "",
            f"Using only occurrence pairs from different Rfam clans on every side, the candidate median was {number(base['base_inclusive_candidate_median_rmsd'])} Å ({base['n_candidates_with_base_inclusive']} words), versus {number(base['base_inclusive_length_random_median_rmsd'])} Å for length-matched random windows and {number(base['base_inclusive_sequence_matched_median_rmsd'])} Å for sequence-matched windows ({base['n_candidates_with_sequence_matched_control']} words with a sequence-matched pool). This whole-nucleotide comparison uses random and sequence-matched windows rather than frequency-matched token words.",
        ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
