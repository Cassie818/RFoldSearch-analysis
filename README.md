<h1><img src=".github/header.PNG" alt="RFoldSearch" height="88" /></h1>

This repository contains the analysis and figure-generation code supporting the **RFoldSearch** manuscript, *RFoldSearch enables fast RNA structure search beyond sequence similarity*: model training and evaluation scripts, benchmark results, plotting scripts and the utilities needed to reproduce the analyses and figures. RFoldSearch is a fast RNA tertiary-structure search method built around **3Dr**, a learned 80-state structural alphabet that encodes local RNA geometry and interaction context without nucleotide identity. It combines indexed 4-mer prefiltering, Structure-BLOSUM token alignment and geometry-aware scoring; an optional coordinate-based reranking mode, **RFoldSearch-TM**, provides greater structural sensitivity.

## Highlights

- **RNA-specific structural representation:** 3Dr captures local tertiary geometry, base-pairing, stacking and spatial interaction context.
- **Structure-aware retrieval:** indexed candidate search, token Smith–Waterman alignment and geometric scoring enable sensitive structural comparison.
- **Retrieval beyond sequence similarity:** RFoldSearch finds structurally related RNAs that nucleotide sequence search misses.
- **Scalable database search:** about 0.47 s per query against 531,888 predicted RNA structures on a single CPU core.
- **Interpretable structural patterns:** recurrent 3Dr words capture conserved local RNA geometry, including fragments of known RNA 3D motifs.

## Installation

Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the versions under which every plotting script was tested. `files/models/db.pt` needs NumPy ≥ 2. Structure preprocessing uses biotite 0.40, which needs NumPy < 2 and therefore a separate environment (see [Dataset construction](#dataset-construction)).

## Manuscript result sources

- Sequence-divergent zone summaries: `files/production/retrieval/sequence_divergent/zone_checks.json`.
- Sequence-filtered family and clan results: `family_perquery.json` and `clan_perquery.json` in the same directory. The `B/N E<=10` key denotes positives reported by BLASTN or nhmmer at E ≤ 10.
- Canonical 313-pair nucleotide-alignment summary: `files/ablation/rfam3d_reference_disjoint_independent_3dseed/rfam3d_reference_alignment_common_313.json`.
- Runtime benchmarks: `files/predicted_runtime/` (see `single_system/README.md` there).

## Repository layout

Scripts are grouped by the part of the manuscript they support.

```
paths.py                          shared path resolution
preprocessing/                    dataset construction and structure preprocessing
src/models/                       tokenizer model, descriptors and training utilities
src/search/                       evaluation metrics and Smith-Waterman helper code
scripts/common/                   plot style shared across figures
scripts/01_model/                 final tokenizer training, ablations, family-disjoint CV
scripts/02_rnasolo_retrieval/     RNAsolo retrieval benchmark, baselines and controls
scripts/03_sequence_divergent/    retrieval of sequence-divergent homologues
scripts/04_alignment_benchmark/   nucleotide-level alignment benchmark against Rfam 3D seeds
scripts/05_runtime/               runtime on the predicted-structure databases
scripts/06_predicted_retrieval/   retrieval on the predicted-structure databases
scripts/07_motifs/                recurrent token words and motif fragments
files/models/                     deployed model artifacts used by analyses
files/production/                 RNAsolo retrieval results
files/ablation/                   ablation, cross-validation and alignment-benchmark results
files/predicted_runtime/          predicted-database runtime results
files/predicted_retrieval/        predicted-database retrieval results
files/motifs/                     motif-analysis tables, DSSR annotations and Fig. 4F/G structures
```

## Dataset construction

The 885-chain non-redundant set is built from RNAsolo per-chain CIF files in three steps. The scripts are in `preprocessing/`; the CIF corpus itself is not tracked. `cdhit_dedup.py` needs `cd-hit-est` on the PATH.

```bash
# 1. Keep only BGSU non-redundant representatives (4.0 A list, v4.38)
python preprocessing/dedup_bgsu.py raw_cifs/ -o bgsu_filtered/

# 2. Remove residual chain-level redundancy at 80% sequence identity
python preprocessing/cdhit_dedup.py bgsu_filtered/ -o nr80/ --identity 0.80

# 3. Backbone coordinates plus base-pair and stacking edges, one JSON per chain
python preprocessing/extract_backbone.py --pdb_dir nr80/ --output nr80.json
```

`add_edges.py` has no command line of its own: `extract_backbone.py` imports `find_base_pair_and_stack_idx` from it and writes the edges into the same records as the coordinates.

## Availability

RFoldSearch is freely available as open-source software, a web server and a REST API.

- **Web server:** [https://www.rfoldsearch.com/](https://www.rfoldsearch.com/)
- **Source code:** [https://github.com/DylanYang/RFoldSearch](https://github.com/DylanYang/RFoldSearch)
- **Analysis and figure reproduction:** [https://github.com/Cassie818/RFoldSearch-analysis](https://github.com/Cassie818/RFoldSearch-analysis)

## License

Apache License 2.0; see [LICENSE](LICENSE).
