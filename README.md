# RFoldseek

Analysis and figure code behind the Results section of the RFoldseek manuscript.

Every script here maps to a specific figure. Ablation results are bundled as
JSON so all figures can be redrawn immediately after cloning; regenerating those
JSONs from scratch additionally requires the large data artifacts described
under [Data](#data).

## Layout

```
RFoldseek                 fast RNA structural search CLI
paths.py                  path resolution (RFOLDSEEK_DATA / RFOLDSEEK_FIGURES)
src/models/               VQ-VAE tokenizer, codebook, descriptors
src/search/               k-mer prefilter, token Smith-Waterman, substitution matrices
scripts/01_train/         final deployed model training
scripts/02_benchmark/     retrieval evaluation and baselines
scripts/03_ablations/     5-fold cross-validated design ablations
scripts/04_figures/       figure generation
files/                    bundled ablation + retrieval results (JSON)
figures/                  output
```

## Quick start

```bash
pip install -r requirements.txt

# Redraw every figure from the bundled results (no large data needed)
for f in scripts/04_figures/plot_fig*.py; do python "$f"; done
```

Figures land in `figures/`. All figures except two redraw from the bundled
results with no extra data. `plot_figS10.py` (substitution matrix) and
`plot_figS11.py` (alphabet states) read the trained model directly and need
`RFOLDSEEK_DATA` (see below).

Bundled results take precedence over `RFOLDSEEK_DATA`, so pointing that variable
at a working tree cannot silently substitute in-progress numbers for the
published ones. Set `RFOLDSEEK_PREFER_EXTERNAL=1` to read results from
`RFOLDSEEK_DATA` instead.

## Search

`RFoldseek` runs the search pipeline:
encode the query to VQ-VAE tokens, run the similar-k-mer-expansion prefilter over
an inverted k-mer index of the target database using the Structure-BLOSUM matrix
(exact per-k-mer self-score threshold, keeping targets with ≥2 hits on a common
diagonal), rerank the candidate pool with token Smith-Waterman, and report the
top-k hits by alignment score, each with a Karlin-Altschul database E-value.

```bash
# search a chain already in the bundled 885-chain database against that database
python RFoldseek --query PDB_00001FFZ_1_A --top_k 10

# encode a preprocessed query structure JSON on the fly, then search
python RFoldseek --query my_rna.json --db files/models/db.pt

# search against a database you built with src/search/build_database.py
python RFoldseek --query my_query_db.pt --query_id chainA --db my_targets.pt
```

`python RFoldseek -h` lists all options (`--db`, `--model`, `--matrix`,
`--top_k`, `--top_n`, `--min_diag`).

Each hit is reported with its Smith-Waterman score and a **database E-value**
(`E = K·m·N·e^(−λS)`: the expected number of chance hits scoring this high in a
target database of `N` tokens, `m` the query length). The E-value length-
normalizes the raw score so significance is comparable across queries of very
different sizes:

```
rank  target                    SW_score    E-value    len
   1  PDB_00001FFK_1_0            203.54   9.4e-246   2706
   2  PDB_00005AN9_1_N            180.02   1.3e-216   1162
   3  PDB_00004V9F_1_0            168.13   7.1e-202   2803
```

The Karlin-Altschul parameters λ, K are calibrated once (random token-pair
Smith-Waterman under the deployed matrix and gap penalties) and bundled in
`files/models/evalue_params.json`, so E-values are available straight after
cloning. Recalibrate — e.g. after changing the substitution matrix or database —
with:

```bash
python src/search/build_evalue_params.py
```

## Data

The deployed model is bundled in the repo under `files/models/`
(tokenizer, Structure-BLOSUM matrix, and tokenised 885-chain database), so
[search](#search) runs straight after cloning. The remaining large inputs —
descriptor tensors and the US-align / RMalign pairwise caches, needed only to
regenerate results from scratch — are not bundled. Point `RFOLDSEEK_DATA` at the
directory containing `files/` and `Splits/`:

```bash
export RFOLDSEEK_DATA=/path/to/RFoldSeek
```

Scripts that need it fail with a clear `FileNotFoundError` if it is unset.

Required layout under `$RFOLDSEEK_DATA`:

```
Splits/labels.tsv                              Rfam family / clan annotation
Splits/pdb_out/{train,val,test}.json           PDB-disjoint split
Splits/pdb_out/{train,val}_desc.pt             descriptor tensors
files/models/db.pt                     tokenized 885-chain database
files/models/sub_matrix.pt             Structure-BLOSUM matrix
files/models/desc_vqvae_best.pt        trained tokenizer
files/models/evalue_params.json        Karlin-Altschul E-value params
files/production/retrieval/usalign_full.cache.jsonl
files/production/retrieval/rmalign_full.cache.jsonl
files/production/chain_sequences.tsv           for the BLASTN baseline
```

## Figure map

| Manuscript figure | Plot script | Computed by | Bundled result |
|---|---|---|---|
| Fig 2A,B | `plot_fig2.py` | `eval_rfam_sens_fig2ab.py` | `per_query_sens_test.json` |
| Fig 2C,D | `plot_fig2.py` | `eval_structural_sens_fig2cd.py` | `structural_tmscore_metrics_test.json` |
| Fig S2 | `plot_figS2.py` | `ablation_figS2_descriptor_blocks.py` | `cv_s2_clan_5fold.json` |
| Fig S3 | `plot_figS3.py` | `ablation_figS3_descriptor_blocks2.py` | `cv_s3_clan_5fold.json` |
| Fig S4 | `plot_figS4.py` | `ablation_figS4_codebook_size.py` | `cv_s4_clan_5fold.json` |
| Fig S5 | `plot_figS5.py` | `ablation_figS5_vqvae_arch.py` | `cv_s5_clan_5fold.json` |
| Fig S6 | `plot_figS6.py` | `ablation_figS6_substitution_matrix.py` | `cv_s6_clan_5fold.json` |
| Fig S7 | `plot_figS7.py` | `ablation_figS7_prefilter.py` | `cv_s7_clan_5fold.json` |
| Fig S8 | `plot_figS8.py` | `ablation_figS8_virtual_centre.py` | `cv_vc_clan_5fold.json` |
| Fig S9 | `plot_figS9.py` | `ablation_figS9_prefilter_scoring.py` | `cv_prefilter_score_5fold.json` |
| Fig S10 | `plot_figS10.py` | `scripts/01_train/train_final_model.py` | needs `RFOLDSEEK_DATA` |
| Fig S11 | `plot_figS11.py` | — (reads the trained model directly) | needs `RFOLDSEEK_DATA` |
| Fig S12 | `plot_figS12.py` | `eval_recall1_figS12.py` | `recall1_rfam_test.json` |

All design ablations (Figs S2–S9) use PDB-disjoint 5-fold cross-validation on the
train+validation partition only, reporting mean ± SD across folds. The test split
is untouched throughout model selection.
