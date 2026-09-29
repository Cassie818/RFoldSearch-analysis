# RFoldseek

This repository contains the analysis and figure-plotting code used for the RFoldseek manuscript.
It is intentionally scoped to the paper analyses: model training/evaluation scripts, bundled result tables,
plotting scripts, and the small helper modules needed to reproduce the reported figures.

## Repository layout

Scripts are grouped by the part of the paper they support. Each figure has one
plotting script that reads only bundled result files; the analyses that produce
those files sit next to it.

```
paths.py                          shared path resolution
preprocessing/                    dataset construction and structure preprocessing
src/models/                       tokenizer model, descriptors and training utilities
src/search/                       evaluation metrics and Smith-Waterman helper code
scripts/common/                   plot style shared across figures (figstyle.py)
scripts/01_model/                 final tokenizer training, ablations, family-disjoint CV
scripts/02_rnasolo_retrieval/     RNAsolo retrieval benchmark and baselines
scripts/03_sequence_divergent/    retrieval of sequence-divergent homologues
scripts/04_alignment_benchmark/   residue-level alignment benchmark against Rfam 3D seeds
scripts/05_runtime/               runtime on the predicted-structure databases
scripts/06_predicted_retrieval/   retrieval on the predicted-structure databases
scripts/07_motifs/                recurrent token words and motif cores
scripts/08_evalue_calibration/    E-value calibration of the experimental and predicted databases
files/models/                     deployed model artifacts used by analyses
files/production/                 RNAsolo retrieval results and the deployed E-value calibrations
files/ablation/                   ablation, cross-validation and alignment-benchmark results
files/predicted_runtime/          predicted-database runtime results
files/predicted_retrieval/        predicted-database retrieval results
files/motifs/                     motif-analysis tables
figures/                          figure outputs
```

## Figures

Plotting scripts write to `figures/` (not tracked). Supplementary figures are
named by content rather than number, so the table gives the mapping.

| Figure | Plotting script | Produced by |
|---|---|---|
| Fig. 1 overview | static `figures/fig1.jpeg` | |
| Fig. 2 RNAsolo retrieval | `02_rnasolo_retrieval/plot_fig2.py` | `eval_rfam_truth.py`, `eval_structure_truth.py`, `structure_truth_seeded.py`, `bootstrap_rfam_truth.py`, `bootstrap_structure_truth.py`, `riboseek_benchmark.py`, `infernal_benchmark.py` |
| Fig. 3 sequence-divergent homologues | `03_sequence_divergent/plot_fig3.py` | `sequence_divergent_rankings.py`, `sequence_divergent_rescore.py` (robustness checks: `sequence_divergent_zone_checks.py`) |
| Fig. 4 runtime | `05_runtime/plot_fig4.py` | bundled timings in `files/predicted_runtime/` |
| Fig. 5 predicted-structure retrieval | `06_predicted_retrieval/plot_fig5.py` | `build_databases.py`, `make_sample.py`, `run_searches.py`, `run_infernal.py`, `run_usalign.py`, `run_riboseek_rs80.py`, `evaluate.py` |
| Fig. 6 motifs | `07_motifs/plot_fig6.py` | `cross_family_kmer_mining.py`, `global_sequence_matched_control.py`, `sequence_matched_control_44_48_62_71.py`, `known_motif_core_dssr_annotation.py`, `bgsu_overlap.py` |
| Ablation overview | static `figures/supp_ablation_overview.jpg` | |
| Descriptor blocks | `01_model/plot_supp_descriptor_blocks.py` | `ablation_descriptor_blocks.py` |
| Descriptor combinations | `01_model/plot_supp_descriptor_combinations.py` | `ablation_descriptor_combinations.py` |
| Codebook size | `01_model/plot_supp_codebook_size.py` | `ablation_codebook_size.py` |
| VQ-VAE architecture | `01_model/plot_supp_vqvae_architecture.py` | `ablation_vqvae_architecture.py` |
| Substitution matrix | `01_model/plot_supp_substitution_matrix.py` | `ablation_substitution_matrix.py` |
| Prefilter seed pattern | `01_model/plot_supp_prefilter_seed.py` | `ablation_prefilter_seed.py` |
| Virtual-centre weights | `01_model/plot_supp_virtual_centre.py` | `ablation_virtual_centre.py` |
| Prefilter scoring | `01_model/plot_supp_prefilter_scoring.py` | `ablation_prefilter_scoring.py` |
| Structure-BLOSUM matrix | `01_model/plot_supp_structure_blosum.py` | `train_final_model.py` |
| Alphabet states | `01_model/plot_supp_alphabet_states.py` | `train_final_model.py` (needs the raw structures) |
| Family-disjoint cross-validation | `01_model/plot_supp_family_disjoint_cv.py` | `cv_family_disjoint.py`, `cv_unlabelled_homology.py`, `cv_labelled_homology.py`, `summarize_cv_family_disjoint.py` |
| Additional retrieval metrics | `02_rnasolo_retrieval/plot_supp_retrieval_metrics.py` | `eval_retrieval_metrics.py`, `bootstrap_retrieval_metrics.py` |
| Quantiser control | `02_rnasolo_retrieval/plot_supp_quantiser_control.py` | `kmeans_quantiser_control.py`, `prefilter_selectivity.py`, `quantiser_structural.py`, `bootstrap_controls.py` |
| RiboSeek-to-RFoldseek attribution | `02_rnasolo_retrieval/plot_supp_attribution_forward.py` | `attribution_paths.py`, `attribution_structural.py`, `riboseek_alphabet_control.py` |
| Attribution controls | `02_rnasolo_retrieval/plot_supp_attribution_controls.py` | as above |
| Alignment benchmark | `04_alignment_benchmark/plot_supp_alignment_benchmark.py` | `fetch_rfam3d_chains.py`, `rfam3d_reference_alignment.py`, `reconstruct_rfam3d_sequence_baseline.py`, `summarize_rfam3d_common_cohort.py` |
| Alignment cases | `04_alignment_benchmark/plot_supp_alignment_cases.py` | `build_alignment_cases.py` |
| Runtime scaling | `05_runtime/plot_supp_runtime_scaling.py` | as Fig. 4 |
| Runtime and query length | `05_runtime/plot_supp_runtime_query_length.py` | as Fig. 4 |
| RiboSeek speed | `05_runtime/plot_supp_riboseek_speed.py` | as Fig. 4 |
| Predicted databases | `05_runtime/plot_supp_predicted_databases.py` | needs the predicted-structure catalogues (not bundled) |
| Predicted retrieval, Rfam truth | `06_predicted_retrieval/plot_supp_predicted_rfam_truth.py` | as Fig. 5 |
| Predicted retrieval, structure truth | `06_predicted_retrieval/plot_supp_predicted_structure_truth.py` | as Fig. 5 |
| E-value calibration | `08_evalue_calibration/plot_supp_evalue_calibration.py` | experimental: `build_experimental_empirical_evalue.py`, `summarize_experimental_evalue_lofo.py`; predicted: `score_predicted_raw_clustermax_shard.py`, `fit_entropy_conditional_evalue.py`, `build_entropy_deployment_candidates.py`, `adjudicate_entropy_conditional_candidates.py`, `summarize_entropy_conditional_evalue.py`, `fit_entropy_deployment_map.py` |

The deployed calibration parameters are
`files/production/evalue_experimental/evalue_params_experimental_empirical.json` and
`files/production/evalue_predicted/evalue_params_predicted_entropy_deployment.json`.
The calibrated search databases themselves are too large for the repository.

## Quick start

```bash
pip install -r requirements.txt

python scripts/02_rnasolo_retrieval/plot_fig2.py
python scripts/03_sequence_divergent/plot_fig3.py
python scripts/05_runtime/plot_fig4.py
python scripts/06_predicted_retrieval/plot_fig5.py
python scripts/07_motifs/plot_fig6.py
```

All plotting scripts run from the bundled files except the two marked above.
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
