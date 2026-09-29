#!/usr/bin/env python3
"""Unified Infernal benchmark entry point for the experimental RNAsolo set.

The benchmark has three logically separate stages:

``run``
    Build one query-specific covariance model per test RNA and write the raw
    rankings to ``infernal_rankings_test.json``.
``score``
    Re-score those rankings using the definitions of Fig. 2 and the retrieval-metrics supplementary figure and
    update the bundled per-query, structural, recall, and PR summaries.
``all``
    Run both stages in sequence.

The lower-level runner and recovery scripts remain available for debugging or
partial reruns, but this file is the reproducible public entry point.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "_infernal_runner.py"
FIG2_SCORER = HERE / "_infernal_retrieval_scorer.py"
FIGS12_SCORER = HERE / "_infernal_metrics_scorer.py"
DEFAULT_JSON = ROOT / "files/production/retrieval/infernal_rankings_test.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Infernal component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_raw() -> None:
    """Run the existing query-specific-CM search stage."""
    module = load_module(RUNNER, "infernal_runner")
    module.main()


def score_results(infernal_json: Path) -> None:
    """Inject one raw ranking file into all main and supplementary summaries."""
    # Fig. 2 scorer accepts the input path as a command-line argument. Keep
    # that interface in one place rather than duplicating its definitions here.
    old_argv = sys.argv
    try:
        sys.argv = [str(FIG2_SCORER), "--infernal-json", str(infernal_json)]
        load_module(FIG2_SCORER, "infernal_fig2_scorer").main()
    finally:
        sys.argv = old_argv

    # the retrieval-metrics supplementary figure uses the same raw ranking file and its own established summary
    # definitions. Its scorer has no arguments by design.
    load_module(FIGS12_SCORER, "infernal_figs12_scorer").main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("run", "score", "all"),
        help="run raw Infernal searches, score existing rankings, or do both",
    )
    parser.add_argument(
        "--infernal-json", type=Path, default=DEFAULT_JSON,
        help="raw per-query ranking JSON used by the score stage",
    )
    args = parser.parse_args()

    if args.stage in ("run", "all"):
        run_raw()
    if args.stage in ("score", "all"):
        score_results(args.infernal_json)


if __name__ == "__main__":
    main()
