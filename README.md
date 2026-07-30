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
top-k hits by alignment score.

```bash
# search a chain already in the bundled 885-chain database against that database
python RFoldseek --query PDB_00001FFZ_1_A --top_k 10

# encode a preprocessed query structure JSON on the fly, then search
python RFoldseek --query my_rna.json --db files/production_new/db.pt

# search against a database you built with src/search/build_database.py
python RFoldseek --query my_query_db.pt --query_id chainA --db my_targets.pt
```

`python RFoldseek -h` lists all options (`--db`, `--model`, `--matrix`,
`--top_k`, `--top_n`, `--min_diag`).

## Data

The deployed model is bundled in the repo under `files/production_new/`
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
files/production_new/db.pt                     tokenized 885-chain database (bundled)
files/production_new/sub_matrix.pt             Structure-BLOSUM matrix (bundled)
files/production_new/desc_vqvae_best.pt        trained tokenizer (bundled)
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


`plot_figS11.py` also writes a companion `token_state_representatives.tsv`
into `figures/` listing the chosen representatives. Because `figures/` is
gitignored it is a local artifact, not a bundled result. Its `frequency` and
`role` columns come from `--profile`
(`files/motifs_v2/token_profile_summary.tsv`), a table that predates the final
retrain, so those two columns refer to the earlier codebook's token indices.
The figure itself does not use the profile and is unaffected.

## Evaluation protocol

Retrieval is evaluated Foldseek-style over the full 885-chain database with
self-hits removed, on the held-out test partition (106 family-evaluable and 79
clan-evaluable query chains), using two metrics:

- **Sens@1FP** — fraction of true positives ranked before the first
  false positive, macro-averaged. Hierarchical labels: at the *family* level
  same-family targets are true positives and same-clan/different-family targets
  are neutral; at the *clan* level same-clan/different-family targets are true
  positives and same-family targets are neutral.
- **Recall@1** — top-ranked non-self target is a true positive,
  with **no neutral set**; clan-level true positives include same-family targets.

The two use different label conventions by design; see the manuscript's
Supplementary Methods.

All design ablations (Figs S2–S9) use PDB-disjoint 5-fold cross-validation on the
train+validation partition only, reporting mean ± SD across folds. The test split
is untouched throughout model selection.

## Deployed configuration

K = 80 codebook, 17-D descriptor (`g_ctx` + `g_nn1`), VQ-VAE with 2-layer encoder
and 4-layer decoder, exact 4-mer inverted-index prefilter with 1-substitution
similar-k-mer expansion, top-100 candidates, token Smith-Waterman reranking with
the Structure-BLOSUM matrix (gap open −2.0, extend −0.5).

Similar k-mers in the prefilter are admitted when the full one-substitution
k-mer score under Structure-BLOSUM clears the threshold (0.75·k), the
Foldseek/MMseqs2 convention. The threshold is evaluated per k-mer from the
query k-mer's own self-score rather than from a per-token cutoff, because the
matrix is diagonal-mean-1 rather than unit-diagonal. The design ablations that
use expansion (Figs S2-S8) admitted neighbours by codebook cosine instead; the
two matrices are compared in Fig S9 and differ by less than the fold-to-fold
variation. Candidate ties at the top-N cutoff are broken deterministically by
target index.

### Substitution matrices

The **deployed** matrix is the **Structure-BLOSUM**, built inside
`scripts/01_train/train_final_model.py` (`build_struct_blosum`) from training
chain pairs whose symmetric (mean-normalised) US-align TM-score reaches 0.5
— `0.5 * (tm_query + tm_target) >= 0.5`, the same TM definition used for the
Fig 2C/D structural ground truth — and shipped as
`files/production_new/sub_matrix.pt`. There is no standalone build script for
it; it is produced as part of training. (The max-normalised variant,
`max(tm_query, tm_target) >= 0.5`, yields indistinguishable retrieval — family
Sens@1FP 0.548, clan 0.223 either way — and was the earlier criterion.)

The two scripts under `src/search/` build the **ablation alternatives**, not
the deployed matrix:

- `build_matrix_codebook_cosine.py` — codebook-cosine matrix.
- `build_matrix_rfam_blosum.py` — Rfam-family-supervised BLOSUM.

Both write to their own filenames and never overwrite the deployed
`sub_matrix.pt`. Two affine-gap settings are involved and are recorded
separately in the deployed matrix's metadata: `construction_gap_*` (−11 / −1,
the traceback that collects aligned token pairs) and `rerank_gap_*` (−2.0 /
−0.5, used with the matrix during Smith-Waterman reranking).

## Baselines

US-align, RMalign, BLASTN, GTalign, RTM-align and RiboSeek. 

## Notes

- `scripts/01_train/train_final_model.py` trains 30 seeds and selects on
  validation Sens@1FP.
- `riboseek_baseline.py` needs Python ≥ 3.9 for the `riboseek` package, while the
  rest of the pipeline was run under Python 3.8. Encoded chains are cached.
