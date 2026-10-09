#!/usr/bin/env python3
"""Fetch chain-level, first-model mmCIF files referenced by Rfam.3d.seed."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PATTERN = re.compile(r"PDB_0*([0-9A-Z]{4})_\d+_(.+)$")


def rfam_chains(path: Path) -> set[str]:
    chains = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith("#=GR"):
                continue
            fields = line.split()
            if len(fields) >= 4 and fields[2].endswith("_SS"):
                chains.add(fields[2][:-3].upper())
    return chains


def local_chains(directories: list[Path]) -> set[str]:
    found = set()
    for directory in directories:
        for path in directory.glob("*.cif"):
            match = PATTERN.match(path.stem.upper())
            if match:
                found.add(f"{match.group(1)}_{match.group(2)}")
    return found


def fetch(chain: str, output_dir: Path, retries: int, source: str) -> dict:
    pdb_id, asym_id = chain.split("_", 1)
    output = output_dir / f"PDB_0000{pdb_id}_1_{asym_id}.cif"
    if output.exists() and output.stat().st_size > 1024:
        return {"chain": chain, "status": "existing", "path": str(output),
                "bytes": output.stat().st_size}
    if source == "rnasolo":
        query = urllib.parse.urlencode({
            "pdbid": pdb_id, "format": "CIF", "chains": asym_id, "models": 1,
        })
        url = f"https://rnasolo.cs.put.poznan.pl/api/query/structure?{query}"
    else:
        query = urllib.parse.urlencode({
            "auth_asym_id": asym_id, "model_nums": 1, "encoding": "cif",
        })
        url = f"https://www.ebi.ac.uk/pdbe/model-server/v1/{pdb_id.lower()}/atoms?{query}"
    error = None
    for attempt in range(retries):
        temporary = output.with_suffix(".cif.part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "RFoldSearch-Rfam3D/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            if len(data) < 1024 or b"_atom_site." not in data:
                raise ValueError(f"response is not a coordinate mmCIF ({len(data)} bytes)")
            temporary.write_bytes(data)
            temporary.replace(output)
            return {"chain": chain, "status": "downloaded", "path": str(output),
                    "bytes": len(data), "url": url}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    return {"chain": chain, "status": "failed", "error": error, "url": url}


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rfam-seed", type=Path)
    source.add_argument(
        "--benchmark-json", type=Path,
        help="Fetch only PDB chains present in a saved reference-alignment benchmark",
    )
    parser.add_argument("--existing-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--source", choices=("pdbe", "rnasolo"), default="pdbe")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.benchmark_json:
        pairs = json.loads(args.benchmark_json.read_text())["pairs"]
        wanted = {row[key].upper() for row in pairs
                  for key in ("query_chain", "target_chain")}
    else:
        wanted = rfam_chains(args.rfam_seed)
    present = local_chains([*args.existing_dir, args.output_dir])
    missing = sorted(wanted - present)
    print(f"Rfam chains={len(wanted)} present={len(present & wanted)} missing={len(missing)}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, chain, args.output_dir, args.retries,
                                   args.source): chain
                   for chain in missing}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{number}/{len(missing)}] {result['chain']}: {result['status']}", flush=True)

    manifest = {
        "source": ("RNAsolo cleaned chain-level CIF API" if args.source == "rnasolo"
                   else "PDBe ModelServer chain-level atom query, first model"),
        "selection_source": str(args.benchmark_json or args.rfam_seed),
        "n_requested_chains": len(wanted),
        "n_already_present": len(present & wanted),
        "results": sorted(results, key=lambda row: row["chain"]),
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    failures = [row for row in results if row["status"] == "failed"]
    print(f"downloaded={sum(row['status'] == 'downloaded' for row in results)} "
          f"failed={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
