#!/usr/bin/env python3
"""US-align adjudication for entropy-conditional E-value candidates."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import torch


def tm_score(binary: Path, query: Path, target: Path) -> float | None:
    try:
        process = subprocess.run(
            [str(binary), str(query), str(target), "-mol", "RNA"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = re.findall(r"TM-score=\s*([0-9.]+)", process.stdout)
    if len(values) < 2:
        return None
    return 0.5 * (float(values[0]) + float(values[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--usalign", type=Path, required=True)
    parser.add_argument("--tm-cache", type=Path)
    parser.add_argument("--tm-threshold", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    tm_cache = {}
    if args.tm_cache:
        cached = json.loads(args.tm_cache.read_text())
        for record in cached["records"]:
            for candidate in record["candidates"]:
                if candidate.get("tm") is not None:
                    tm_cache[(record["query"], int(candidate["target_index"]))] = float(candidate["tm"])
    database = torch.load(args.db, map_location="cpu", weights_only=False)
    ids = [Path(str(entry["id"])).stem for entry in database["entries"]]
    completed = 0
    reused = 0
    unresolved = 0
    for record in data["records"]:
        query_path = args.structures / f"{record['query']}.pdb"
        for candidate in record["candidates"]:
            target = ids[candidate["target_index"]]
            target_path = args.structures / f"{target}.pdb"
            tm = candidate.get("tm", tm_cache.get((record["query"], int(candidate["target_index"]))))
            if tm is None:
                tm = tm_score(args.usalign, query_path, target_path)
                completed += 1
            else:
                reused += 1
            candidate["target"] = target
            candidate["tm"] = tm
            candidate["background"] = tm is not None and tm < args.tm_threshold
            unresolved += int(tm is None)
            if completed and completed % 100 == 0:
                print(f"{completed} new candidates adjudicated", flush=True)
    data["tm_threshold"] = args.tm_threshold
    data["n_usalign"] = completed + reused
    data["n_usalign_new"] = completed
    data["n_usalign_reused"] = reused
    data["n_unresolved"] = unresolved
    data["n_background"] = sum(
        sum(candidate["background"] for candidate in record["candidates"])
        for record in data["records"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({
        "n_usalign": completed + reused,
        "n_usalign_new": completed,
        "n_usalign_reused": reused,
        "n_unresolved": unresolved,
        "n_background": data["n_background"],
    }, indent=2))


if __name__ == "__main__":
    main()
