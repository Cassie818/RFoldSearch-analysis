# RFoldseek

This repository contains the analysis and figure-generation code used for the
RFoldseek manuscript. It is intentionally scoped to the paper analyses: model
training/evaluation scripts, bundled result tables, plotting scripts, and the
small helper modules needed to reproduce the reported figures.

This is not the standalone RFoldseek search package. Generated figure outputs
and local split files are ignored by Git.

## Repository layout

```
paths.py                  shared path resolution
preprocessing/            RNA structure preprocessing helpers
src/models/               tokenizer model, descriptors, and training utilities
src/search/               evaluation metrics and Smith-Waterman helper code
scripts/01_train/         final tokenizer training script
scripts/02_benchmark/     RNAsolo retrieval evaluations and baselines
scripts/03_ablations/     model-design and alphabet-control analyses
scripts/04_figures/       manuscript and supplementary figure plotting scripts
scripts/05_motif/         recurrent token-word and motif-core analyses
files/models/             deployed model artifacts used by analyses
files/production/         bundled retrieval results for Fig. 2 and Fig. S12
files/ablation/           bundled ablation and alphabet-control results
files/motifs/             bundled motif-analysis tables for Fig. 4
```

Plotting scripts write regenerated figures to `figures/`; split files should be supplied through
the external data directory described below.

## Quick start

```bash
pip install -r requirements.txt

# Redraw figures from bundled result files
for f in scripts/04_figures/plot_fig*.py; do python "$f"; done
```

Most figures redraw directly from bundled JSON/TSV/PT artifacts. Two
supplementary plots read the deployed tokenizer directly:

```bash
python scripts/04_figures/plot_figS10.py
python scripts/04_figures/plot_figS11.py
```

Fig. 4 can be redrawn from the bundled motif tables with:

```bash
python scripts/04_figures/plot_fig4.py
```

## Data

Small analysis artifacts needed for plotting are bundled under `files/`. Larger
intermediate inputs, such as descriptor tensors, split files, and full pairwise
alignment caches used to regenerate benchmark outputs from scratch, are not
tracked. To rerun those analyses, point `RFOLDSEEK_DATA` to a local data tree:

```bash
export RFOLDSEEK_DATA=/path/to/RFoldSeek
```

Expected external layout:

```
Splits/labels.tsv
Splits/pdb_out/{train,val,test}.json
Splits/pdb_out/{train,val}_desc.pt
files/production/retrieval/usalign_full.cache.jsonl
files/production/retrieval/rmalign_full.cache.jsonl
files/production/chain_sequences.tsv
```

Bundled files in this repository are preferred by default for reproducible
plotting. Set `RFOLDSEEK_PREFER_EXTERNAL=1` only when you explicitly want a
script to read matching results from `RFOLDSEEK_DATA`.

## Figure map

| Figure | Plot script | Analysis script(s) | Bundled result |
|---|---|---|---|
| Fig. 2A,B | `scripts/04_figures/plot_fig2.py` | `scripts/02_benchmark/eval_rfam_sens_fig2ab.py` | `files/production/retrieval/per_query_sens_test.json`, `bootstrap_fig2ab_band.json` |
| Fig. 2C,D | `scripts/04_figures/plot_fig2.py` | `scripts/02_benchmark/eval_structural_sens_fig2cd.py` | `files/production/retrieval/structural_tmscore_metrics_test.json`, `bootstrap_fig2cd_results.json` |
| Fig. 4 | `scripts/04_figures/plot_fig4.py` | `scripts/05_motif/*.py` | `files/motifs/` |
| Fig. S2 | `scripts/04_figures/plot_figS2.py` | `scripts/03_ablations/ablation_figS2_descriptor_blocks.py` | `files/ablation/cv_s2_clan_5fold.json` |
| Fig. S3 | `scripts/04_figures/plot_figS3.py` | `scripts/03_ablations/ablation_figS3_descriptor_blocks2.py` | `files/ablation/cv_s3_clan_5fold.json` |
| Fig. S4 | `scripts/04_figures/plot_figS4.py` | `scripts/03_ablations/ablation_figS4_codebook_size.py` | `files/ablation/cv_s4_clan_5fold.json` |
| Fig. S5 | `scripts/04_figures/plot_figS5.py` | `scripts/03_ablations/ablation_figS5_vqvae_arch.py` | `files/ablation/cv_s5_clan_5fold.json` |
| Fig. S6 | `scripts/04_figures/plot_figS6.py` | `scripts/03_ablations/ablation_figS6_substitution_matrix.py` | `files/ablation/cv_s6_clan_5fold.json` |
| Fig. S7 | `scripts/04_figures/plot_figS7.py` | `scripts/03_ablations/ablation_figS7_prefilter.py` | `files/ablation/cv_s7_clan_5fold.json` |
| Fig. S8 | `scripts/04_figures/plot_figS8.py` | `scripts/03_ablations/ablation_figS8_virtual_centre.py` | `files/ablation/cv_vc_clan_5fold.json` |
| Fig. S9 | `scripts/04_figures/plot_figS9.py` | `scripts/03_ablations/ablation_figS9_prefilter_scoring.py` | `files/ablation/cv_prefilter_score_5fold.json` |
| Fig. S10 | `scripts/04_figures/plot_figS10.py` | `scripts/01_train/train_final_model.py` | reads `files/models/desc_vqvae_best.pt` |
| Fig. S11 | `scripts/04_figures/plot_figS11.py` | direct model/codebook inspection | reads `files/models/desc_vqvae_best.pt` and `files/models/db.pt` |
| Fig. S12 | `scripts/04_figures/plot_figS12.py` | `scripts/02_benchmark/eval_recall1_figS12.py` | `files/production/retrieval/recall1_rfam_test.json` |

All design ablations use PDB-disjoint five-fold cross-validation on the
train-plus-validation partition. The held-out test partition is used only for
the final retrieval evaluations.
