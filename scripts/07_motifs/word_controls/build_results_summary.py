"""Build the manuscript-facing summary exclusively from generated outputs."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR / "results_summary.md")
    args = parser.parse_args()
    frequency = metrics(OUT_DIR / "frequency_matched_random_words_summary.tsv")
    base = metrics(OUT_DIR / "base_inclusive_rmsd_summary.tsv")
    pairing = metrics(OUT_DIR / "pairing_pattern_stratification_summary.tsv")
    bgsu = metrics(OUT_DIR / "bgsu_random_window_background_summary.tsv")

    base_complete = base.get("status") == "complete"
    if base_complete:
        base_key_result = (
            f"- Base-inclusive RMSD: Atom-set-A candidate median RMSD was "
            f"{number(base['atom_set_A_candidate_median_rmsd'])} Å, versus "
            f"{number(base['atom_set_A_length_random_median_rmsd'])} Å for length-matched random windows "
            f"and {number(base['atom_set_A_sequence_matched_median_rmsd'])} Å for sequence-matched windows "
            f"({base['n_candidates_with_atom_set_A']} candidate words). Source: "
            "`base_inclusive_rmsd_summary.tsv`."
        )
        base_comparison_lines = [
            "### Coordinate-set comparison",
            "",
            "| Coordinates | Candidate median RMSD (Å) | Length-matched random (Å) | Sequence-matched (Å) |",
            "|---|---:|---:|---:|",
            f"| Five anchors | {number(base['five_anchor_candidate_all_pair_median_rmsd'])} | {number(base['five_anchor_length_random_median_rmsd'])} | {number(base['five_anchor_sequence_matched_median_rmsd'])} |",
            f"| Atom set A | {number(base['atom_set_A_candidate_median_rmsd'])} | {number(base['atom_set_A_length_random_median_rmsd'])} | {number(base['atom_set_A_sequence_matched_median_rmsd'])} |",
            "",
            "Source: `base_inclusive_rmsd_summary.tsv`.",
        ]
        base_results_text = (
            f"> With the identity-independent base-inclusive atom set A, recurrent-word occurrences retained a "
            f"median RMSD of {number(base['atom_set_A_candidate_median_rmsd'])} Å, substantially below both "
            f"length-matched random windows ({number(base['atom_set_A_length_random_median_rmsd'])} Å) and "
            f"sequence-matched windows ({number(base['atom_set_A_sequence_matched_median_rmsd'])} Å)."
        )
        base_methods_heading = "**Base-inclusive RMSD.**"
    else:
        base_key_result = (
            f"- Base-inclusive analysis status: `{base['status']}`; mmCIF files found: "
            f"{base['n_mmcif_files_found']}. Source: `base_inclusive_rmsd_summary.tsv`."
        )
        base_results_text = (
            "The base-inclusive RMSD todo cannot yet be replaced by a numerical sentence because the required "
            "mmCIF directory was absent when the analysis was run."
        )
        base_methods_heading = "**Base-inclusive RMSD (to run once mmCIF input is restored).**"
        base_comparison_lines = []

    lines = [
        "# Recurrent 3Dr word controls: results summary",
        "",
        "Every numerical value below is read from a generated TSV file.",
        "",
        "## Key results",
        "",
        f"- Frequency-matched control: candidate all-pair median RMSD was {number(frequency['candidate_all_pair_median_rmsd'])} Å (P10–P90 {number(frequency['candidate_all_pair_p10_rmsd'])}–{number(frequency['candidate_all_pair_p90_rmsd'])}) versus {number(frequency['frequency_matched_control_median_rmsd'])} Å (P10–P90 {number(frequency['frequency_matched_control_p10_rmsd'])}–{number(frequency['frequency_matched_control_p90_rmsd'])}) for matched non-candidate words. {percent(frequency['fraction_candidates_below_matched_control'])} of candidates were below their matched control (Wilcoxon P = {frequency['wilcoxon_candidate_vs_matched_p']}). Source: `frequency_matched_random_words_summary.tsv`.",
        f"- Shuffled-word control: {frequency['n_candidates_with_existing_shuffled_word']} candidates had an observed shuffled word; its median RMSD was {number(frequency['shuffled_control_median_rmsd'])} Å and the paired Wilcoxon P value was {frequency['wilcoxon_candidate_vs_shuffled_p']}. Source: `frequency_matched_random_words_summary.tsv`.",
        f"- Pairing strata: category (a) contained {pairing['category_a_n_words']} words and the remaining categories contained {pairing['nonstandard_after_removing_a_n_words']} words. After removing (a), candidate all-pair median RMSD was {number(pairing['nonstandard_after_removing_a_candidate_all_pair_median_rmsd'])} Å versus {number(pairing['nonstandard_after_removing_a_matched_control_median_rmsd'])} Å for matched controls; {percent(pairing['nonstandard_after_removing_a_fraction_candidate_below_control'])} of candidates were lower (Wilcoxon P = {pairing['nonstandard_after_removing_a_wilcoxon_p']}). Source: `pairing_pattern_stratification_summary.tsv`.",
        f"- BGSU background: the candidate-word median overlap fraction was {percent(bgsu['candidate_word_median_overlap_fraction'])}, compared with a random-window median of {percent(bgsu['background_overlap_median_fraction'])} (P10–P90 {percent(bgsu['background_overlap_p10_fraction'])}–{percent(bgsu['background_overlap_p90_fraction'])}). Source: `bgsu_random_window_background_summary.tsv`.",
        base_key_result,
        "",
        *base_comparison_lines,
        "",
        "### Pairing-stratum details",
        "",
        "| Category | Words | Cross-family median RMSD (Å) | Matched-control median RMSD (Å) | Median BGSU overlap | Words with BGSU denominator |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| ({key}) {pairing[f'category_{key}_name']} | {pairing[f'category_{key}_n_words']} | {number(pairing[f'category_{key}_cross_family_median_rmsd'])} | {number(pairing[f'category_{key}_matched_control_median_rmsd'])} | {percent(pairing[f'category_{key}_bgsu_overlap_median_fraction'])} | {pairing[f'category_{key}_n_words_with_bgsu_denominator']} |"
            for key in "abcd"
        ],
        "",
        "Source: `pairing_pattern_stratification_summary.tsv`.",
        "",
        "## Interpretation",
        "",
        "The frequency-matched analysis weakens the claim that recurrent cross-family words are unusually geometrically coherent merely because they pass the candidate filter: non-candidate words matched by occurrence frequency had lower, not higher, all-pair RMSD. The shuffled-word comparison also did not show a significant paired difference. The conservative conclusion is therefore that a 3Dr word identifies a reproducible local geometry, but recurrence across families does not by itself make that geometry tighter than other equally frequent token words.",
        "",
        "Removing standard-helical words does not rescue the original comparison: non-helical candidates remained higher than their frequency-matched controls. In contrast, enrichment within BGSU-annotated motifs remains supported relative to random windows sampled from the same BGSU-covered chains.",
        "" if not base_complete else "Base-inclusive coordinates support a narrower claim that identical recurrent words preserve whole-nucleotide geometry relative to random windows, even though candidate selection does not outperform equally frequent non-candidate token words under the five-anchor comparison.",
        "",
        "## Replacement text for Results todos",
        "",
        f"> Across {frequency['n_candidates']} recurrent words, the median all-pair five-anchor RMSD was {number(frequency['candidate_all_pair_median_rmsd'])} Å, compared with {number(frequency['frequency_matched_control_median_rmsd'])} Å for non-candidate words matched by occurrence-frequency bin; only {percent(frequency['fraction_candidates_below_matched_control'])} of candidates were lower than their matched controls (paired Wilcoxon P = {frequency['wilcoxon_candidate_vs_matched_p']}). Thus, cross-family recurrence did not confer tighter geometry than frequency-matched token words. The same conclusion held after excluding standard cis-paired helices ({number(pairing['nonstandard_after_removing_a_candidate_all_pair_median_rmsd'])} versus {number(pairing['nonstandard_after_removing_a_matched_control_median_rmsd'])} Å; {percent(pairing['nonstandard_after_removing_a_fraction_candidate_below_control'])} lower).",
        "",
        f"> Candidate words overlapped BGSU hairpin or internal-loop annotations with a median fraction of {percent(bgsu['candidate_word_median_overlap_fraction'])}, compared with {percent(bgsu['background_overlap_median_fraction'])} for random four-nucleotide windows sampled from BGSU-covered chains.",
        "",
        base_results_text,
        "",
        "## Methods text",
        "",
        f"> **Frequency-matched word controls.** We enumerated all four-token words occurring at least eight times in the 3Dr database and excluded the {frequency['n_candidates']} recurrent candidates. For each candidate, one non-candidate word was sampled from the same occurrence-frequency bin ({frequency['frequency_bins']}) in each of {frequency['n_repeats']} replicates using random seed {frequency['random_seed']}. Candidate and control geometry was compared using the all-pair median RMSD over the five stored anchors, with Kabsch alignment and RMSD imported from `cross_family_kmer_mining.py`. As a secondary control, token order was permuted and retained when the permuted word occurred at least eight times. Paired differences were tested with a two-sided Wilcoxon signed-rank test.",
        "",
        f"> **Pairing-pattern stratification.** Pairing state and cis/trans orientation were taken from the pipeline Biotite annotations. Each occurrence was encoded as a four-position paired/unpaired pattern together with its partner-index geometry. Words were assigned their modal occurrence signature and divided into standard cis-paired antiparallel helices, other fully paired words, partly unpaired words, and fully unpaired words. Pair annotations were available for {pairing['n_occurrences_with_pair_annotations']} enumerated candidate occurrences; {pairing['n_occurrences_missing_pair_annotations']} occurrences lacked these arrays in the compact inputs and were excluded from modal-signature estimation.",
        "",
        f"> **BGSU background.** Four-nucleotide windows were sampled uniformly from the {bgsu['n_bgsu_covered_chains']} database chains covered by the downloaded BGSU Hairpin and Internal Loop Motif Atlas files. Each replicate contained {bgsu['n_candidate_occurrences_on_bgsu_covered_chains']} windows, matching the number of candidate occurrences on those chains, and overlap was defined as any residue intersection, as in `bgsu_overlap.py`. Sampling was repeated {bgsu['n_repeats']} times with random seed {bgsu['random_seed']}.",
        "",
        f"> {base_methods_heading} Atom set A comprises P, OP1, OP2, O5′, C5′, C4′, O4′, C3′, O3′, C2′, O2′ and C1′ plus N9/C4/C8 for purines or N1/C2/C6 for pyrimidines. Occurrences with missing required atoms or non-mappable modified residues are excluded with reason-specific counts. Candidate estimates use at most {base.get('max_pairs_per_comparison', 'NA')} occurrence pairs per word; each of the {base.get('n_repeats', 'NA')} length- and sequence-matched control replicates uses at most {base.get('max_pairs_per_control_replicate', 'NA')} pairs, with random seed {base.get('random_seed', 'NA')}.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
