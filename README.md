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
preprocessing/            dataset construction and structure preprocessing
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

**Why two redundancy steps.** BGSU's non-redundant list groups structures at
the IFE (Integrated Functional Element) level, not the chain level. Once
multi-chain IFEs are expanded into their constituent chains, the same RNA
species reappears across IFE classes, so a chain-level pass is still needed.
`cdhit_dedup.py` runs CD-HIT-EST at `-c 0.80` with `-aS 0.80 -aL 0.80`, which
requires 80% alignment coverage of *both* sequences, so a short chain that is
merely contained in a longer one is not clustered away. Cluster
representatives are chosen deterministically: longest chain, ties broken by
best resolution, then by filename.

**Residue identification is shared.** `cdhit_dedup.py`, `extract_backbone.py`
and `add_edges.py` use the same `RNA_BASES` set and the same 19-entry
`MODIFIED_TO_CANONICAL` map, so clustering and coordinate extraction apply one
definition of what counts as an RNA residue. The two differ in what counts as
*resolved*: clustering records a residue if the CIF has any atom for it, while
coordinate extraction requires C3', C4' and O4' to all be present. For 854 of
the 885 chains the two lengths agree exactly; across the whole set the
coordinate side is shorter by 0.1% of residues, concentrated in a few chains
with unresolved backbone atoms.

**Sequence redundancy is not structural redundancy.** The set is
non-redundant at 80% sequence identity, which does not bound structural
similarity: 1357 of the 361244 chain pairs (0.38%) reach a symmetric US-align
TM-score of 0.90 or higher. Retrieval numbers on this set should be read with
that in mind — every method benefits from it equally, but the absolute
sensitivities are higher than they would be on a structurally deduplicated
set.

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

All design ablations use PDB-disjoint five-fold cross-validation on the
train-plus-validation partition. The held-out test partition is used only for
the final retrieval evaluations.
