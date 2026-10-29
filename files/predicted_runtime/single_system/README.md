# Single-system runtime re-benchmark (2026-10-29)

US-align, RTM-align, RMalign, Infernal and Infernal-fast were timed directly on the reference
system (Intel Xeon Platinum 8470Q, one core per single-threaded job, up to 18 concurrent jobs),
and GTalign `--speed=13` directly on the RTX 4090. This replaces the earlier cross-system
conversion of the E5-2683 v3 timings.

- Coordinate aligners: 300 RhoFold+ queries at 1k, 5k, 10k and 83,156 targets
  (`../rhofold/coord_8470.json`, with per-query times; RTM-align entries carry `n_skip`).
- Infernal (`cmsearch`) and Infernal-fast (`--rfam`): 300 queries at all six sizes
  (`../rhofold/infernal*_{n}.json`, with per-query times and hit counts).
- GTalign-fast: 300 queries at all six sizes (`../rhofold/gtalign_fast_{n}.json`). A default-mode
  re-run at 83,156 gave 4.37 s per query, against 4.33 s in `bench_83156.json`.

Binaries, command lines, queries (`queries_300.txt`) and target lists are the same as in the
earlier runs. `run_single.py` and `gt13_rho.py` are the drivers, and `build_single.py` builds the
summaries from the per-job results.
