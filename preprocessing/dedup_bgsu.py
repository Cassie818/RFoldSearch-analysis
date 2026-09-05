"""
keep only files matching a BGSU non-redundant representative.
Input formats supported:
  - URL (default: current 4.0Å list)
  - Local CSV path (downloaded from http://rna.bgsu.edu/rna3dhub/nrlist)

Usage:
    python dedup_bgsu.py /path/to/input_cifs/ -o /path/to/output_dir/
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# BGSU nrlist v4.38 (22 April 2026), 4.0 Å resolution threshold.
# The versioned release URL is preferred for reproducibility; if it
# returns 404, fall back to the /current/ endpoint.
# Note: BGSU's URL pattern is /download/rna/<version>/<resolution>/csv
# (the /rna/ segment is required for versioned downloads; /current/
# uses a different shorter path).
_VERSIONED_URL = "https://rna.bgsu.edu/rna3dhub/nrlist/download/rna/4.38/4.0A/csv"
_CURRENT_URL   = "https://rna.bgsu.edu/rna3dhub/nrlist/download/current/4.0A/csv"
DEFAULT_NRLIST = _VERSIONED_URL

# Single-chain RNA: e.g. "1MJI|1|D"
SINGLE_REP_RE = re.compile(r"^([0-9A-Za-z]{4})\|(\d+)\|([A-Za-z0-9]+)$")

# Our filename pattern: PDB_0000XXXX_M_C.cif
FILE_RE = re.compile(r"PDB_0000([0-9A-Za-z]{4})_(\d+)_([A-Za-z0-9]+)\.cif$")


def load_nrlist(path_or_url: str) -> list:
    """Return a list of (representative, members_str) tuples per class."""
    if path_or_url.startswith(("http://", "https://")):
        print("Fetching nrlist: %s" % path_or_url)
        try:
            data = urllib.request.urlopen(path_or_url, timeout=60).read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404 and path_or_url == _VERSIONED_URL:
                print("  Versioned URL returned 404, falling back to /current/")
                data = urllib.request.urlopen(_CURRENT_URL, timeout=60).read().decode()
            else:
                raise
        lines = data.splitlines()
    else:
        with open(path_or_url) as f:
            lines = f.read().splitlines()
    classes = []
    for row in csv.reader(lines):
        if len(row) >= 3:
            classes.append((row[1], row[2]))
        elif len(row) >= 2:
            classes.append((row[1], row[1]))
    return classes


def _parse_rep_part(part: str):
    """Parse one 'PDB|model|chain' fragment. Return (pdb, model, chain) or None."""
    m = SINGLE_REP_RE.match(part.strip())
    if not m:
        return None
    return (m.group(1).lower(), m.group(2), m.group(3))


def reps_to_keys(classes: list) -> tuple:
    """Convert BGSU equivalence classes to a set of chain keys.

    Following the dataset construction in RhoFold+ (Shen et al., 2024),
    we focus on single-chain RNAs and keep ALL members of each
    single-chain equivalence class (not only the representative).
    Class-level non-redundancy is then re-asserted at the chain level
    by CD-HIT-EST clustering downstream.

    Concretely:
      * Multi-chain IFE classes (representative contains '+') are
        skipped entirely, since their members are co-folded RNA
        complexes that don't fit a single-chain RNA pipeline.
      * For each remaining class, every member chain
        (PDB|model|chain) is added as a key.

    Returns (keys, n_multi_skipped, n_unparseable_classes).
    """
    keys = set()
    n_multi_skipped, n_unparseable = 0, 0
    for rep, members_str in classes:
        if "+" in rep:
            n_multi_skipped += 1
            continue
        added = 0
        for member in members_str.split(","):
            member = member.strip()
            if not member or "+" in member:
                continue
            k = _parse_rep_part(member)
            if k:
                keys.add(k)
                added += 1
        if added == 0:
            n_unparseable += 1
    return keys, n_multi_skipped, n_unparseable


def file_key(name: str):
    m = FILE_RE.search(name)
    if not m:
        return None
    return (m.group(1).lower(), m.group(2), m.group(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory of CIF files")
    parser.add_argument("-o", "--output_dir",
                        help="Output directory")
    parser.add_argument("--nrlist", default=None,
                        help="Path to a downloaded nrlist CSV")
    parser.add_argument("--nrlist_url", default=DEFAULT_NRLIST,
                        help="URL to fetch nrlist from (used if --nrlist absent)")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink instead of copy")
    args = parser.parse_args()

    src = args.nrlist or args.nrlist_url
    classes = load_nrlist(src)
    rep_keys, n_multi_skipped, n_unparseable = reps_to_keys(classes)
    print("nrlist source: %s" % src)
    print("  total classes:                  %d" % len(classes))
    print("  multi-chain classes (skipped):  %d" % n_multi_skipped)
    print("  single-chain classes (kept):    %d" % (len(classes) - n_multi_skipped))
    print("  unparseable classes:            %d" % n_unparseable)
    print("  member chain keys:              %d" % len(rep_keys))

    in_dir = Path(args.input_dir)
    cifs = sorted(in_dir.glob("*.cif"))
    print()
    print("Input dir: %s   (%d files)" % (in_dir, len(cifs)))

    # Match
    keep, drop, unparseable = [], [], []
    for p in cifs:
        k = file_key(p.name)
        if k is None:
            unparseable.append(p)
            continue
        if k in rep_keys:
            keep.append(p)
        else:
            drop.append(p)

    print()
    print("=" * 60)
    print("L4 BGSU nrlist intersection")
    print("=" * 60)
    print("  Kept (representatives):  %d" % len(keep))
    print("  Dropped:       %d" % len(drop))
    print("  Unparseable filenames:   %d" % len(unparseable))
    if unparseable:
        for p in unparseable[:5]:
            print("    e.g. %s" % p.name)

    # Sample of what's kept vs dropped
    if drop:
        print()
        print("  Sample dropped (first 5):")
        for p in drop[:5]:
            print("    %s" % p.name)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for p in keep:
            dst = out / p.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.symlink:
                dst.symlink_to(p.resolve())
            else:
                shutil.copy2(p, dst)
        print()
        print("  Wrote %d files to %s" % (len(keep), out))


if __name__ == "__main__":
    main()
