"""Chain-level redundancy reduction with CD-HIT-EST.

BGSU's non-redundant list groups RNA structures into equivalence
classes at the IFE (Integrated Functional Element) level, not the
chain level. After dedup_bgsu.py expands multi-chain IFEs into their
constituent chains (5S/5.8S/LSU/SSU rRNAs, snRNAs, etc.), the same
RNA species can appear in multiple IFE classes and reintroduce
redundancy at the chain level.

This script removes that residual redundancy:
  1. Extract RNA sequence from each CIF (ATOM records, A/C/G/U).
  2. Cluster sequences with CD-HIT-EST at a configurable identity
     threshold (default 0.80; -n 5 word size; -r 0 forward strand only;
     -aS/-aL 0.80 require reciprocal alignment coverage
     for RNA; -d 0 keep full headers).
  3. For each cluster, pick the representative as: longest chain;
     ties broken by best (lowest) resolution parsed from the CIF;
     ties further broken by filename (alphabetical) for full
     reproducibility.
  4. Copy the chosen CIFs to the output directory.

Requires: cd-hit-est on PATH (`brew install cd-hit` or
`conda install -c bioconda cd-hit`).

Usage:
    python cdhit_dedup.py /path/to/bgsu_filtered/ -o /path/to/out/ \\
        --identity 0.80
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RNA_BASES = {"A", "C", "G", "U"}

# Map common modified residues to their canonical base for clustering
# only — the original CIF is preserved on disk.
MODIFIED_TO_CANONICAL = {
    "PSU": "U", "5MU": "U", "4SU": "U", "H2U": "U", "UR3": "U",
    "5MC": "C", "OMC": "C", "5BU": "U", "OMG": "G", "1MG": "G",
    "2MG": "G", "M2G": "G", "7MG": "G", "YG": "G", "YYG": "G",
    "1MA": "A", "6MZ": "A", "MA6": "A", "I": "A",
}


def _chain_id_from_filename(path: Path) -> str | None:
    """Extract chain ID from RNAsolo filename PDB_0000XXXX_M_C.cif."""
    parts = path.stem.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else None


DNA_BASES = {"DA", "DC", "DG", "DT", "DU"}


def extract_rna_sequence(cif_path: Path) -> str:
    """Extract the RNA sequence from a CIF file in residue order.

    Pure manual mmCIF parsing — no third-party structure library. We
    locate the `_atom_site` loop, read the column-header block to
    discover where `label_comp_id`, `label_asym_id`, and
    `label_seq_id` (or `auth_seq_id`) live, then iterate the data
    rows. Residues are dedup-ed by (chain, seq_id) so multiple atoms
    in the same residue collapse to one symbol.

    Modified ribonucleosides are mapped to canonical bases via
    MODIFIED_TO_CANONICAL; DNA residues are skipped; anything else
    is encoded as 'N'.
    """
    try:
        f = open(cif_path)
    except Exception as e:
        print(f"  [open fail] {cif_path.name}: {e}", file=sys.stderr)
        return ""

    try:
        line_iter = iter(f)
        cols = []
        first_data = None
        for line in line_iter:
            stripped = line.strip()
            if stripped.startswith("loop_"):
                # Look ahead: is this the atom_site loop?
                header_lines = []
                for hl in line_iter:
                    if hl.strip().startswith("_atom_site."):
                        header_lines.append(hl)
                        break
                    if hl.strip().startswith("_") or hl.strip() == "":
                        # Not the atom_site loop; skip until next loop_.
                        header_lines = None
                        break
                if header_lines is None:
                    continue
                # Read remaining _atom_site.* headers + first data line.
                cols.append(header_lines[0].strip().split(".", 1)[1].strip())
                for hl in line_iter:
                    s = hl.strip()
                    if s.startswith("_atom_site."):
                        cols.append(s.split(".", 1)[1].strip())
                    else:
                        first_data = hl
                        break
                break

        if not cols or first_data is None:
            return ""

        # Pick column indices we care about. Fall back gracefully.
        try:
            i_group = cols.index("group_PDB")
        except ValueError:
            i_group = 0
        try:
            i_comp = cols.index("label_comp_id")
        except ValueError:
            i_comp = 5
        try:
            i_chain = cols.index("label_asym_id")
        except ValueError:
            i_chain = 6
        if "label_seq_id" in cols:
            i_seq = cols.index("label_seq_id")
        elif "auth_seq_id" in cols:
            i_seq = cols.index("auth_seq_id")
        else:
            i_seq = 8

        seen = []  # ordered list of (chain, seq_id, comp_id)
        seen_set = set()

        def handle(line: str) -> bool:
            """Process one data line; return False at end-of-loop."""
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                return False
            parts = s.split()
            if len(parts) <= max(i_group, i_comp, i_chain, i_seq):
                return True
            if parts[i_group] not in ("ATOM", "HETATM"):
                return True
            chain = parts[i_chain]
            seq_id = parts[i_seq]
            comp = parts[i_comp].strip().upper()
            key = (chain, seq_id)
            if key not in seen_set:
                seen_set.add(key)
                seen.append((chain, seq_id, comp))
            return True

        if not handle(first_data):
            pass
        else:
            for line in line_iter:
                if not handle(line):
                    break
    finally:
        f.close()

    if not seen:
        return ""

    # If the CIF holds multiple chains, restrict to the chain inferred
    # from the filename (RNAsolo per-chain files are single-chain in
    # practice, but be defensive).
    chains_present = {c for c, _, _ in seen}
    if len(chains_present) > 1:
        inferred = _chain_id_from_filename(cif_path)
        target = inferred if inferred in chains_present else sorted(chains_present)[0]
        seen = [t for t in seen if t[0] == target]
        if not seen:
            return ""

    seq_chars = []
    for _, _, comp in seen:
        if comp in RNA_BASES:
            seq_chars.append(comp)
        elif comp in MODIFIED_TO_CANONICAL:
            seq_chars.append(MODIFIED_TO_CANONICAL[comp])
        elif comp in DNA_BASES:
            continue
        else:
            seq_chars.append("N")
    return "".join(seq_chars)


_RES_PATTERNS = [
    re.compile(r"_refine\.ls_d_res_high\s+([0-9.?]+)"),
    re.compile(r"_reflns\.d_resolution_high\s+([0-9.?]+)"),
    re.compile(r"_em_3d_reconstruction\.resolution\s+([0-9.?]+)"),
]


def parse_resolution(cif_path: Path) -> float:
    """Parse the structure resolution from a CIF; return inf if absent.

    Lower is better. NMR / unknown structures get +inf so they sort
    last in tiebreaks.
    """
    try:
        text = cif_path.read_text()
    except Exception:
        return float("inf")
    for pat in _RES_PATTERNS:
        m = pat.search(text)
        if m:
            val = m.group(1)
            if val and val != "?":
                try:
                    return float(val)
                except ValueError:
                    continue
    return float("inf")


def write_fasta(records: list, fasta_path: Path) -> None:
    with open(fasta_path, "w") as fh:
        for name, seq in records:
            fh.write(">%s\n%s\n" % (name, seq))


def run_cdhit(fasta_in: Path, fasta_out: Path, identity: float) -> None:
    # Word size rule for cd-hit-est: 0.8-1.0 -> 8-10; we use 5 to allow
    # 0.80 threshold (n must satisfy: 0.80 -> n in [4,5]).
    if identity >= 0.95:
        n = 10
    elif identity >= 0.90:
        n = 8
    elif identity >= 0.88:
        n = 7
    elif identity >= 0.85:
        n = 6
    elif identity >= 0.80:
        n = 5
    else:
        n = 4
    cmd = [
        "cd-hit-est",
        "-i", str(fasta_in),
        "-o", str(fasta_out),
        "-c", str(identity),
        "-n", str(n),
        "-aS", "0.80",  # at least 80% of the shorter sequence aligned
        "-aL", "0.80",  # at least 80% of the longer sequence aligned
        "-r", "0",   # forward strand only. Reverse-complement clustering
                     # can merge unrelated RNAs and is inappropriate for
                     # chain-level RNA structure deduplication.
        "-d", "0",   # keep full FASTA header
        "-M", "0",   # no memory cap
        "-T", "0",   # all available threads
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_clstr(clstr_path: Path) -> dict:
    """Return {cluster_id: [member_name, ...]} from cd-hit .clstr."""
    clusters: dict = {}
    cur = None
    with open(clstr_path) as fh:
        for line in fh:
            if line.startswith(">Cluster"):
                cur = int(line.strip().split()[1])
                clusters[cur] = []
            else:
                # e.g. "0\t150nt, >1abc_1_A... *"
                m = re.search(r">(\S+?)\.\.\.", line)
                if m and cur is not None:
                    clusters[cur].append(m.group(1))
    return clusters


def pick_representative(members, name_to_path, name_to_len, name_to_res):
    """Longest sequence; ties → best (lowest) resolution; ties → name."""
    def sort_key(name):
        return (-name_to_len[name], name_to_res[name], name)
    return sorted(members, key=sort_key)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="Directory of CIF files (post BGSU)")
    ap.add_argument("-o", "--output_dir", required=True,
                    help="Output directory for cluster representatives")
    ap.add_argument("--identity", type=float, default=0.80,
                    help="CD-HIT-EST identity threshold (default 0.80)")
    ap.add_argument("--workdir", default=None,
                    help="Working dir for FASTA & .clstr (default: tmp)")
    ap.add_argument("--symlink", action="store_true",
                    help="Symlink instead of copy")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    cifs = sorted(in_dir.glob("*.cif"))
    print("Input dir: %s   (%d files)" % (in_dir, len(cifs)))
    if not cifs:
        sys.exit("No CIF files found.")

    # 1. Extract sequences
    records = []
    name_to_path: dict = {}
    name_to_len: dict = {}
    name_to_res: dict = {}
    skipped = 0
    for p in cifs:
        seq = extract_rna_sequence(p)
        if len(seq) == 0:
            skipped += 1
            continue
        name = p.stem  # e.g. PDB_00001abc_1_A
        records.append((name, seq))
        name_to_path[name] = p
        name_to_len[name] = len(seq)
        name_to_res[name] = parse_resolution(p)
    print("Extracted sequences: %d   (skipped empty: %d)" % (len(records), skipped))

    # 2. Run CD-HIT-EST
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="cdhit_"))
    workdir.mkdir(parents=True, exist_ok=True)
    fasta_in = workdir / "input.fasta"
    fasta_out = workdir / "clusters.fasta"
    write_fasta(records, fasta_in)
    run_cdhit(fasta_in, fasta_out, args.identity)

    # 3. Parse clusters and pick representatives
    clusters = parse_clstr(Path(str(fasta_out) + ".clstr"))
    print("CD-HIT-EST clusters: %d" % len(clusters))

    chosen = []
    for cid, members in clusters.items():
        rep = pick_representative(members, name_to_path, name_to_len, name_to_res)
        chosen.append(rep)
    print("Representatives chosen: %d" % len(chosen))

    # 4. Copy out
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        src = name_to_path[name]
        dst = out / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
    print("Wrote %d files to %s" % (len(chosen), out))
    print("Workdir kept at: %s" % workdir)


if __name__ == "__main__":
    main()
