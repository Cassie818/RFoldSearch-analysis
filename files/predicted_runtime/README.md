# Runtime benchmarks

This directory contains the runtime measurements used for the manuscript's runtime Results, main runtime figure, supplementary scaling figures and runtime table. The current RFoldSearch measurements use version 1.0.2. The RiboSeek baselines use version 0.2.1.

All reported runs used the same 300 queries per database on an Intel Xeon Platinum 8470Q with one CPU core and one thread per query. Database construction and index construction were excluded from per-query search time.

For the NuFold RiboSeek subset runs below full scale, `n_requested` is the plotted target count and `n_database` also includes the 300 encoded query entries. The source JSON preserves both values. At full scale the queries are already members of the 531,888-entry database, so no entries are added. This provenance detail should be considered when interpreting the smallest-size scaling points.

## Final figure data

`rhofold/` contains the RhoFold+ ladder up to 83,156 structures. `nufold/` contains the NuFold ladder up to 531,888 structures.

| Method | RhoFold+ files | NuFold files |
|---|---|---|
| RFoldSearch 1.0.2 (`top_n=500`) | `kernel_timing_{n}.json` | `nufold_timing_{n}.json` |
| RiboSeek 0.2.1 RS-20 | `riboseek_rs20_{n}.json` | `nufold_riboseek_rs20_{n}.json` |
| RiboSeek 0.2.1 RS-80 | `riboseek_rs80_{n}.json` | `nufold_riboseek_rs80_{n}.json` |
| RFoldSearch-TM | `rhofold_tm_timing_83156.json` | `nufold_tm_timing_{n}.json` |
| GTalign | `gtalign_fast_{n}.json`, `bench_{n}.json` | `nufold_gtalign_{n}.json`, `nufold_gtalign13_{n}.json` |
| Infernal | `infernal_{n}.json`, `infernal_fast_{n}.json` | not run |
| BLASTN | `bench_{n}.json` | `nufold_blastn_{n}.json` |
| US-align, RTM-align and RMalign | `coord_8470.json` | not run |

RFoldSearch-TM was not re-timed for version 1.0.2 because its runtime is dominated by US-align reranking. Other unchanged baselines retain their October measurements.

At full scale, the mean per-query times are:

| Database | RFoldSearch 1.0.2 | RiboSeek RS-20 0.2.1 | RiboSeek RS-80 0.2.1 |
|---|---:|---:|---:|
| RhoFold+ (83,156) | 0.587 s | 7.081 s | 0.103 s |
| NuFold (531,888) | 0.308 s | 6.172 s | 0.0855 s |

## Raw version 1.0.2 output

`v102_raw/results_v102_top500/` contains the RFoldSearch 1.0.2 runtime ladders. `v102_raw/results_v102_mem/` contains the full-database timing and peak-memory runs. The `.hwm` files report `VmHWM` in kB. The RS-20 peak-memory measurements were obtained separately on a 180 GB instance of the same CPU type and were 94,634,428 kB for RhoFold+ and 97,494,320 kB for NuFold.

The benchmark driver and reference launch script are `scripts/05_runtime/bench_speed_v102.py` and `scripts/05_runtime/run_v102.sh`. The launch script records the original server paths and CPU affinity used for the run.

The launch script calls the server-side `rs021_ref_sizes.py` driver for the RiboSeek runs. That external driver was not included in the supplied runtime archive; its resulting JSON measurements and the exact invocation are retained here.
