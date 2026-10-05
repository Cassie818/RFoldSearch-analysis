# Recurrent 3Dr word controls: results summary

Every numerical value below is read from a generated TSV file.

## Key results

- Frequency-matched control: candidate all-pair median RMSD was 0.542 Å (P10–P90 0.324–1.060) versus 0.308 Å (P10–P90 0.280–0.347) for matched non-candidate words. 8.6% of candidates were below their matched control (Wilcoxon P = 9.8841743046906e-311). Source: `frequency_matched_random_words_summary.tsv`.
- Shuffled-word control: 1080 candidates had an observed shuffled word; its median RMSD was 0.582 Å and the paired Wilcoxon P value was 0.3879094363892127. Source: `frequency_matched_random_words_summary.tsv`.
- Pairing strata: category (a) contained 986 words and the remaining categories contained 1066 words. After removing (a), candidate all-pair median RMSD was 0.599 Å versus 0.308 Å for matched controls; 12.5% of candidates were lower (Wilcoxon P = 2.0493608249617758e-152). Source: `pairing_pattern_stratification_summary.tsv`.
- BGSU background: the candidate-word median overlap fraction was 85.7%, compared with a random-window median of 62.2% (P10–P90 61.1%–63.1%). Source: `bgsu_random_window_background_summary.tsv`.
- Base-inclusive RMSD: the candidate median was 0.769 Å, versus 2.964 Å for length-matched random windows and 2.839 Å for sequence-matched windows (2052 candidate words). Source: `base_inclusive_rmsd_summary.tsv`.

### Coordinate-set comparison

| Coordinates | Candidate median RMSD (Å) | Length-matched random (Å) | Sequence-matched (Å) |
|---|---:|---:|---:|
| Five anchors | 0.542 | 2.312 | 2.247 |
| Base-inclusive atoms | 0.769 | 2.964 | 2.839 |

Source: `base_inclusive_rmsd_summary.tsv`.

### Pairing-stratum details

| Category | Words | Cross-family median RMSD (Å) | Matched-control median RMSD (Å) | Median BGSU overlap | Words with BGSU denominator |
|---|---:|---:|---:|---:|---:|
| (a) standard_helix_PPPP_cis_antiparallel | 986 | 0.538 | 0.309 | 50.0% | 514 |
| (b) all_paired_nonstandard | 309 | 0.598 | 0.306 | 100.0% | 126 |
| (c) partly_unpaired_1_to_3_U | 715 | 0.734 | 0.309 | 100.0% | 268 |
| (d) all_unpaired_UUUU | 42 | 2.357 | 0.310 | 100.0% | 25 |

Source: `pairing_pattern_stratification_summary.tsv`.

## Interpretation

The frequency-matched analysis weakens the claim that recurrent cross-family words are unusually geometrically coherent merely because they pass the candidate filter: non-candidate words matched by occurrence frequency had lower, not higher, all-pair RMSD. The shuffled-word comparison also did not show a significant paired difference. The conservative conclusion is therefore that a 3Dr word identifies a reproducible local geometry, but recurrence across families does not by itself make that geometry tighter than other equally frequent token words.

Removing standard-helical words does not rescue the original comparison: non-helical candidates remained higher than their frequency-matched controls. In contrast, enrichment within BGSU-annotated motifs remains supported relative to random windows sampled from the same BGSU-covered chains.
Base-inclusive coordinates support a narrower claim that identical recurrent words preserve whole-nucleotide geometry relative to random windows, even though candidate selection does not outperform equally frequent non-candidate token words under the five-anchor comparison.

## Replacement text for Results todos

> Across 2052 recurrent words, the median all-pair five-anchor RMSD was 0.542 Å, compared with 0.308 Å for non-candidate words matched by occurrence-frequency bin; only 8.6% of candidates were lower than their matched controls (paired Wilcoxon P = 9.8841743046906e-311). Thus, cross-family recurrence did not confer tighter geometry than frequency-matched token words. The same conclusion held after excluding standard cis-paired helices (0.599 versus 0.308 Å; 12.5% lower).

> Candidate words overlapped BGSU hairpin or internal-loop annotations with a median fraction of 85.7%, compared with 62.2% for random four-nucleotide windows sampled from BGSU-covered chains.

> With the identity-independent base-inclusive atom set, recurrent-word occurrences retained a median RMSD of 0.769 Å, substantially below both length-matched random windows (2.964 Å) and sequence-matched windows (2.839 Å).

## Methods text

> **Frequency-matched word controls.** We enumerated all four-token words occurring at least eight times in the 3Dr database and excluded the 2052 recurrent candidates. For each candidate, one non-candidate word was sampled from the same occurrence-frequency bin (8-15;16-31;>=32) in each of 25 replicates using random seed 260810. Candidate and control geometry was compared using the all-pair median RMSD over the five stored anchors, with Kabsch alignment and RMSD imported from `cross_family_kmer_mining.py`. As a secondary control, token order was permuted and retained when the permuted word occurred at least eight times. Paired differences were tested with a two-sided Wilcoxon signed-rank test.

> **Pairing-pattern stratification.** Pairing state and cis/trans orientation were taken from the pipeline Biotite annotations. Each occurrence was encoded as a four-position paired/unpaired pattern together with its partner-index geometry. Words were assigned their modal occurrence signature and divided into standard cis-paired antiparallel helices, other fully paired words, partly unpaired words, and fully unpaired words. Pair annotations were available for 41911 enumerated candidate occurrences; 0 occurrences lacked these arrays in the compact inputs and were excluded from modal-signature estimation.

> **BGSU background.** Four-nucleotide windows were sampled uniformly from the 175 database chains covered by the downloaded BGSU Hairpin and Internal Loop Motif Atlas files. Each replicate contained 3560 windows, matching the number of candidate occurrences on those chains, and overlap was defined as any residue intersection, as in `bgsu_overlap.py`. Sampling was repeated 25 times with random seed 260810.

> **Base-inclusive RMSD.** The base-inclusive atom set comprises P, OP1, OP2, O5′, C5′, C4′, O4′, C3′, O3′, C2′, O2′ and C1′ plus N9/C4/C8 for purines or N1/C2/C6 for pyrimidines. Occurrences with missing required atoms or non-mappable modified residues are excluded with reason-specific counts. Candidate estimates use at most 400 occurrence pairs per word; each of the 25 length- and sequence-matched control replicates uses at most 25 pairs, with random seed 260810.
