# RFoldseek

This repository contains the analysis and figure-plotting code used for the RFoldseek manuscript.
It is intentionally scoped to the paper analyses: model training/evaluation scripts, bundled result tables,
plotting scripts, and the small helper modules needed to reproduce the reported figures.

## Overview

Scripts are grouped by the part of the paper they support. Each figure has one
plotting script that reads only bundled result files; the analyses that produce
those files sit next to it.

```
paths.py                          shared path resolution
preprocessing/                    dataset construction and structure preprocessing
src/models/                       tokenizer model, descriptors and training utilities
src/search/                       evaluation metrics and Smith-Waterman helper code
scripts/common/                   plot style shared across figures
scripts/01_model/                 final tokenizer training, ablations, family-disjoint CV
scripts/02_rnasolo_retrieval/     RNAsolo retrieval benchmark and baselines
scripts/03_sequence_divergent/    retrieval of sequence-divergent homologues
scripts/04_alignment_benchmark/   nucleotide-level alignment benchmark against Rfam 3D seeds
scripts/05_runtime/               runtime on the predicted-structure databases
scripts/06_predicted_retrieval/   retrieval on the predicted-structure databases
scripts/07_motifs/                recurrent token words and motif cores
files/models/                     deployed model artifacts used by analyses
files/production/                 RNAsolo retrieval results
files/ablation/                   ablation, cross-validation and alignment-benchmark results
files/predicted_runtime/          predicted-database runtime results
files/predicted_retrieval/        predicted-database retrieval results
files/motifs/                     motif-analysis tables
```

## Quick start

```bash
pip install -r requirements.txt

python scripts/02_rnasolo_retrieval/plot_fig2.py
python scripts/03_sequence_divergent/plot_fig3.py
python scripts/05_runtime/plot_fig4.py
python scripts/06_predicted_retrieval/plot_fig5.py
python scripts/07_motifs/plot_fig6.py
```

## Manuscript result sources

- Sequence-divergent zone summaries: `files/production/retrieval/sequence_divergent/zone_checks.json`.
- Sequence-filtered family and clan results: `family_perquery.json` and `clan_perquery.json` in the same directory. The `B/N E<=10` key denotes positives reported by BLASTN or nhmmer at $E \leq 10$.
- Canonical 313-pair nucleotide-alignment summary: `files/ablation/rfam3d_reference_disjoint_independent_3dseed/rfam3d_reference_alignment_common_313.json`.

All plotting scripts run from the bundled files, except the alphabet-states and
predicted-database supplements, which need the raw structures.
Analyses that need data outside the repository (raw structures, the full
training and validation splits, external binaries) read them from the directory
given by `RFOLDSEEK_DATA`; leave it unset when regenerating figures, so that the
bundled results are used. External binaries are located through
`RFOLDSEEK_BLAST_BIN`, `RFOLDSEEK_HMMER_BIN` and `RFOLDSEEK_INFERNAL_BIN`
(directories; empty means `PATH`).

## Dataset construction

The 885-chain non-redundant set is built from RNAsolo per-chain CIF files in
three steps. The scripts are in `preprocessing/`; the CIF corpus itself is not
tracked. `cdhit_dedup.py` needs `cd-hit-est` on the PATH.

```bash
# 1. Keep only BGSU non-redundant representatives (4.0 A list, v4.38)
python preprocessing/dedup_bgsu.py raw_cifs/ -o bgsu_filtered/

# 2. Remove residual chain-level redundancy at 80% sequence identity
python preprocessing/cdhit_dedup.py bgsu_filtered/ -o nr80/ --identity 0.80

# 3. Backbone coordinates plus base-pair and stacking edges, one JSON per chain
python preprocessing/extract_backbone.py --pdb_dir nr80/ --output nr80.json
```

`add_edges.py` has no command line of its own: `extract_backbone.py` imports
`find_base_pair_and_stack_idx` from it and writes the edges into the same
records as the coordinates.
