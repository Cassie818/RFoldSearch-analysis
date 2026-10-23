# Recurrent 3Dr word controls: corrected results summary

Every numerical value below is read from a generated TSV. Primary RMSD comparisons use only pairs from different Rfam clans on both sides; a family without a clan is its own clan.

## Key results

- Cross-clan RMSD: recurrent candidates had median 0.616 Å, versus 0.982 Å for occurrence-frequency-standardised non-candidate words; 78.5% of candidates were below the descriptive matched-control median. A unique-word permutation test stratified by occurrence-frequency bin gave P = 1.00e-04 (2054 candidate and 513 control words; 10000 permutations).
- Control homology audit: among the 3390 non-candidate words occurring at least eight times, 43.2% had no cross-family pair and 84.9% had no cross-clan pair. For the median control word, 100.0% of its cross-family pairs fell within one clan. The primary pool retained 513 words (by frequency bin 8-15:453;16-31:50;>=32:10); their median number of sampled cross-clan pairs was 11, versus 35 for candidates.
- Minimum-pair sensitivity: requiring at least 5 cross-clan pairs per control word (pool 8-15:421;16-31:42;>=32:8) gave a control median of 0.889 Å; 75.9% of candidates were lower (stratified permutation P = 1.00e-04).
- Cross-family diagnostic (homologous same-clan pairs allowed): 0.586 versus 0.370 Å; 13.6% of candidates were lower.
- Pairing-matched analysis: after excluding standard helices, 1068 recurrent words had median cross-clan RMSD 0.754 Å, versus 1.005 Å for controls matched by frequency bin and modal pairing category; 67.9% were lower (unique-word stratified permutation P = 1.00e-04).
- BGSU, word-level statistic: under the permissive ≥1-residue rule, the observed median per-word overlap was 85.7%, versus 66.7% under the word-size-matched null (P10–P90 66.7%–71.4%; Monte Carlo P = 0.001).
- BGSU, pooled diagnostic: the same candidate occurrences gave 50.0%, below the random-window rate of 62.1%. The word-level and pooled summaries answer different questions and must not be compared as if they were the same statistic.
- BGSU sensitivity: with ≥3/4 residues required, the pooled candidate fraction was 16.2% versus 31.3% for random windows; with 4/4 residues it was 8.1% versus 13.5%. Per-word medians were 0.0% and 0.0% (Monte Carlo P = 1.000 and 1.000).

### Pairing-stratum comparison

| Category | Candidate words | Candidate cross-clan median RMSD (Å) | Category-matched control (Å) | Candidate lower | Holm-adjusted permutation P |
|---|---:|---:|---:|---:|---:|
| (a) standard_helix_PPPP_cis_antiparallel | 986 | 0.548 | 0.603 | 68.5% | 0.045 |
| (b) all_paired_nonstandard | 310 | 0.663 | 0.766 | 62.3% | 0.212 |
| (c) partly_unpaired_1_to_3_U | 716 | 0.799 | 1.085 | 70.7% | 4.00e-04 |
| (d) all_unpaired_UUUU | 42 | 2.424 | 2.757 | 61.9% | 0.212 |

## Interpretation

When comparisons are restricted to different-clan occurrence pairs, recurrent words are geometrically tighter than occurrence-frequency-matched non-candidate words, and the direction holds when control words must have more cross-clan pairs and when controls are additionally matched by pairing category. The cross-family diagnostic uses a broader eligible control pool and is therefore not a fixed-pool estimate of the clan restriction.

The BGSU result is statistic-dependent. A typical sparsely observed word is more likely than its size-matched null to have a high permissive overlap fraction, driven by non-helical words, but the pooled occurrence fraction is lower than background and stricter overlap definitions show no enrichment. The defensible conclusion is therefore limited to word-level enrichment under the ≥1-residue rule; broad motif enrichment is not supported.

## Suggested Results text

> Comparing only occurrence pairs from different Rfam clans, recurrent 3Dr words had a median five-anchor RMSD of 0.616 Å, compared with 0.982 Å for occurrence-frequency-standardised non-candidate words; 78.5% of candidates were lower than the descriptive matched-control median. A unique-word permutation test stratified by occurrence-frequency bin gave P = 1.00e-04. Among non-helical words with controls additionally stratified by modal pairing category, the medians were 0.754 and 1.005 Å (stratified permutation P = 1.00e-04). Thus, recurrent words were lower in RMSD than the matched cross-clan token-word background.

> Using the permissive criterion that at least one of four residues intersected a BGSU hairpin or internal-loop annotation, the median overlap fraction across candidate words was 85.7%, compared with 66.7% in a null distribution that matched each word's number of observable occurrences (Monte Carlo P = 0.001). In contrast, the pooled candidate occurrence fraction was 50.0%, below the 62.1% background window rate, and no enrichment was detected when at least three or all four residues were required to overlap. We therefore interpret this as limited word-level enrichment under the permissive overlap definition.

## Updated Methods text

> **Frequency-matched word controls.** We enumerated all non-candidate four-token words occurring at least eight times. Different Rfam families within one clan are treated as related, and 84.9% of these words had no occurrence pair from different clans; the primary control pool therefore retained only words with at least 1 cross-clan pair (families without a clan were treated as singleton clans). For descriptive frequency standardisation, 25 controls were sampled with replacement for each candidate from the same occurrence-frequency bin (8-15;16-31;>=32) using random seed 260810. Candidate and control RMSDs were calculated only between occurrences assigned to different clans, using at most 400 pairs per word. Statistical inference treated each candidate and control word as one independent unit: word labels were permuted 10000 times within occurrence-frequency bins while preserving the observed candidate and control counts, using the candidate-count-weighted mean of stratum-specific median RMSD differences as the test statistic. Two-sided empirical P values used the plus-one correction. A sensitivity analysis required at least 5 cross-clan pairs per control word. Cross-family, all-pair and token-order permutation results were retained as diagnostics rather than primary tests.

> **Pairing-pattern stratification.** Candidate and eligible control words were assigned to standard helical, fully paired non-standard, partially unpaired or fully unpaired categories from their modal Biotite pairing pattern. Descriptive controls were matched jointly by occurrence-frequency bin and modal pairing category. Statistical inference permuted unique word labels within the same frequency-bin-by-category strata using 10000 permutations. Category-specific P values were adjusted across the four categories by the Holm method. Both sides used only different-clan occurrence pairs. Pair annotations were available for 41927 candidate occurrences; 0 occurrences were excluded for missing annotations.

> **BGSU background and sensitivity analysis.** For every candidate word observed on BGSU-covered chains, each null replicate sampled the same number of random four-nucleotide windows as that word's observed denominator. The median overlap fraction across words was then calculated, preserving the candidate word as the inferential unit. Sampling used 1000 replicates, replacement, random seed 260810 and 15974 eligible windows from 175 chains. Overlap thresholds of at least one, at least three and all four residues within the same BGSU motif occurrence were evaluated. Pooled occurrence fractions were reported separately as descriptive sensitivity checks.

## Shared-universe controls (Fig. 6A)

All eight distributions in Fig. 6A use one universe: Rfam-labelled occurrences with unique full residue-key mapping and complete base-inclusive atoms. On the same sampled occurrence pairs, the five-anchor medians were 0.616 Å for candidate words, 0.773 Å for frequency-matched words, 2.314 Å for length-random windows and 2.200 Å for sequence-matched windows; the base-inclusive medians were 0.873, 1.095, 2.948 and 2.822 Å. Candidate versus frequency-matched words: five-anchor P = 1.00e-04, base-inclusive P = 1.00e-04 (unique-word permutation within occurrence-frequency bins, 10000 permutations; 241 control words).

> **Shared-universe controls.** Fig. 6A compares candidate words with three controls on one set of windows: Rfam-labelled four-token windows whose residues map to unique full residue keys (PDB entry, model, chain, nucleotide, author number and insertion code) and carry complete base-inclusive atoms. Each comparison was computed twice on the same sampled occurrence pairs, with the five RFoldSearch anchor atoms and with base-inclusive atoms (12 backbone and sugar atoms plus N9, C4, C8 for purines or N1, C2, C6 for pyrimidines). Only occurrence pairs from different Rfam clans were used. Frequency-matched controls were 25 non-candidate words drawn with replacement from the candidate's occurrence-frequency bin among words occurring at least eight times in this universe (a candidate with fewer occurrences was compared with the 8–15 bin); length-random controls replaced each occurrence with a random window, and sequence-matched controls with a window of the same four-nucleotide sequence and a different token word. Candidate and frequency-control RMSDs used at most 400 pairs per word, random and sequence-matched replicates at most 25. Candidates were compared with the frequency-control pool by a unique-word permutation test within frequency bins (two-sided, plus-one correction).

> **DSSR signatures.** All DSSR pairing records of each core nucleotide were retained. A GNRA fragment matched when cWW and tSH were among the records at positions 1 and 2 and tSH and cWW at positions 5 and 6 (positions 3 and 4 unconstrained); a T-loop fragment matched when tWH, tWS and cWW were among the records at positions 1–3 (position 4 unconstrained). Additional pairing records did not prevent a match.

## Base-inclusive coordinate diagnostic

Using only occurrence pairs from different Rfam clans on every side, the candidate median was 0.873 Å (2054 words), versus 2.948 Å for length-matched random windows and 2.822 Å for sequence-matched windows (2054 words with a sequence-matched pool). This whole-nucleotide comparison uses random and sequence-matched windows rather than frequency-matched token words.
