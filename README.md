# RFoldseek

This repository contains the analysis and figure-plotting code used for the RFoldseek manuscript.
It is intentionally scoped to the paper analyses: model training/evaluation scripts, bundled result tables,
plotting scripts, and the small helper modules needed to reproduce the reported figures.

## Repository layout

```
paths.py                  shared path resolution
preprocessing/            dataset construction and structure preprocessing
src/models/               tokenizer model, descriptors, and training utilities
src/search/               evaluation metrics and Smith-Waterman helper code
scripts/01_model/         final tokenizer training and model ablations
scripts/02_rnasolo_benchmark/ RNAsolo retrieval evaluations and baselines
scripts/03_token_alignment/  Rfam 3D reference-alignment analyses
scripts/04_predicted_runtime/ predicted-database runtime analyses
scripts/05_predicted_retrieval/ predicted-database retrieval analyses
scripts/06_motifs/         recurrent token-word and motif-core analyses
files/models/             deployed model artifacts used by analyses
files/production/         bundled retrieval and calibration results
files/ablation/           bundled ablation and alignment results
files/predicted_runtime/  bundled predicted-database runtime results
files/predicted_retrieval/ bundled predicted-database retrieval results
files/motifs/             bundled motif-analysis tables for Fig. 5
```

## Quick start

```bash
pip install -r requirements.txt

# Main figures, from bundled result files
python scripts/02_rnasolo_benchmark/plot_fig2.py
python scripts/04_predicted_runtime/plot_fig3.py
python scripts/05_predicted_retrieval/plot_fig4.py
python scripts/06_motifs/plot_fig5.py
```

Supplementary plots are grouped with the analyses that produce them. For
example, the model-ablation figures read:

```bash
python scripts/01_model/plot_figS10.py
python scripts/01_model/plot_figS11.py
```

The Rfam 3D alignment benchmark and its case-study plot are run from
`scripts/03_token_alignment/`. Experimental-database E-value calibration
scripts are kept with the RNAsolo analyses in
`scripts/02_rnasolo_benchmark/`.

Use `run` to regenerate only the raw rankings or `score` to update the Fig. 2
and Fig. S12 summaries from an existing ranking file.

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
