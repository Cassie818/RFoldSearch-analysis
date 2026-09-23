#!/usr/bin/env python3
"""Re-run zero-second Infernal failures with the manuscript's -L 0.01 setting.

This is a small recovery utility for the raw Fig. 2 run. It imports the
original runner so that CM construction, secondary-structure annotation and
cmsearch parsing remain identical, then replaces only successfully recovered
records in the local production copy of the rankings.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = ROOT / "scripts/02_rnasolo_benchmark/_infernal_runner.py"
DEFAULT_JSON = ROOT / "files/production/retrieval/infernal_rankings_test.json"


def load_runner(path: Path, calib_l: str):
    spec = importlib.util.spec_from_file_location("infernal_fig2_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.CALIB_L = calib_l
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calib-l", default="0.01",
                        help="cmcalibrate random-sequence length in Mb")
    parser.add_argument("--cif-dir", type=Path, default=None,
                        help="directory containing QUERY_ID.cif recovery inputs")
    args = parser.parse_args()

    runner = load_runner(args.runner, args.calib_l)
    data = json.load(open(args.json))
    failed = [
        query for query, result in data.items()
        if float(result.get("seconds", 0.0)) == 0.0 and not result.get("ranking")
    ]
    cif_dir = args.cif_dir if args.cif_dir is not None else Path(runner.NR80)
    jobs = [(query, str(cif_dir / (query + ".cif"))) for query in failed]
    print(f"Re-running {len(jobs)} zero-second queries with "
          f"cmcalibrate -L {args.calib_l}", flush=True)

    recovered = 0
    # Each task spends its time in external Infernal executables, so threads
    # provide parallelism without requiring the dynamically imported runner
    # functions to be pickleable.
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(runner.one_query, job) for job in jobs]
        for future in as_completed(futures):
            query, ranking, seconds, status = future.result()
            print(f"{query}: {status} ({seconds:.1f} s; {len(ranking)} hits)", flush=True)
            if status == "ok":
                data[query] = {"ranking": ranking, "seconds": seconds}
                recovered += 1
                # Checkpoint after every expensive calibration so an interrupted
                # recovery run resumes only the records that remain unfinished.
                with open(args.json, "w") as handle:
                    json.dump(data, handle)

    with open(args.json, "w") as handle:
        json.dump(data, handle)
    print(f"Recovered {recovered}/{len(jobs)} queries; wrote {args.json}")


if __name__ == "__main__":
    main()
