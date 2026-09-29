"""Sequence-level homology between unlabelled and labelled CV-pool chains.

Feeds the unlabelled-chain filter of cv_family_disjoint.py (--split family).
Writes one TSV row per detected unlabelled chain -> labelled chain / family:

  BLASTN  -task blastn, E < 0.01, searched in both directions
          (unlabelled vs labelled db, labelled vs unlabelled db)
  nhmmer  E < 0.01, both directions
  cmscan  relaxed scan of unlabelled chains against all of Rfam (no GA
          cut-offs, no --rfam filter tuning, E < 0.1); a hit counts toward
          the fold whose test families include the hit's family

Binaries: RFOLDSEEK_BLAST_BIN, RFOLDSEEK_HMMER_BIN, RFOLDSEEK_INFERNAL_BIN
(directories; empty = PATH). Rfam CM: data/ref/rfam/Rfam.cm under DATA_ROOT.
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, BLAST_BIN, INFERNAL_BIN, SRC_SEARCH, resolve

_sys.path.insert(0, SRC_SEARCH)
SEQS = resolve("files/production/chain_sequences.tsv")
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
OUT = _os.path.join(_REPO, "files/ablation/cv_family_disjoint_unlabelled_seqhits.tsv")
_os.chdir(DATA_ROOT)
import csv, json, re, subprocess, tempfile
from pathlib import Path
import torch

from prefilter_rfam_benchmark import attach_labels, load_labels

HMMER_BIN = _os.environ.get("RFOLDSEEK_HMMER_BIN", "")
SEQ_E = 0.01
CMSCAN_E = 0.1
CPU = str(_os.cpu_count() or 4)
RFAM_CM = "data/ref/rfam/Rfam.cm"


def _bin(d, name):
    return _os.path.join(d, name) if d else name


db = torch.load("files/production/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
fams = attach_labels(ents, load_labels("Splits/labels.tsv"))


def idl(p):
    return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]


pool = [id2i[c] for c in idl("Splits/pdb_out/train.json") + idl("Splits/pdb_out/val.json") if c in id2i]
tag = {d: _os.path.basename(str(ents[d]["id"])).removesuffix(".cif") for d in pool}
seqs = {r["chain_id"]: r["sequence"] for r in csv.DictReader(open(SEQS), delimiter="\t")}
missing = [tag[d] for d in pool if tag[d] not in seqs]
assert not missing, f"no sequence for {missing[:5]}"


def clean(s):
    return re.sub(r"[^ACGU]", "N", s.upper().replace("T", "U"))


unlab = [d for d in pool if not fams[d]]
lab = [d for d in pool if fams[d]]
work = Path(tempfile.mkdtemp(prefix="cv_hom_"))
fa = {}
for name, ds in (("unlab", unlab), ("lab", lab)):
    fa[name] = work / f"{name}.fa"
    with open(fa[name], "w") as f:
        for d in ds:
            f.write(f">{tag[d]}\n{clean(seqs[tag[d]])}\n")
lab_fam = {tag[d]: fams[d] for d in lab}
unlab_tags = {tag[d] for d in unlab}
rows = []

# BLASTN, both directions
for name in ("unlab", "lab"):
    subprocess.run([_bin(BLAST_BIN, "makeblastdb"), "-in", str(fa[name]), "-dbtype", "nucl",
                    "-out", str(work / name)], check=True, capture_output=True)
for q, t in (("unlab", "lab"), ("lab", "unlab")):
    out = work / f"blast_{q}_{t}.tsv"
    subprocess.run([_bin(BLAST_BIN, "blastn"), "-task", "blastn", "-query", str(fa[q]), "-db", str(work / t),
                    "-evalue", str(SEQ_E), "-dust", "no", "-num_threads", CPU, "-max_target_seqs", "1000",
                    "-outfmt", "6 qseqid sseqid evalue bitscore", "-out", str(out)],
                   check=True, capture_output=True)
    for line in open(out):
        a, b, e, _ = line.rstrip("\n").split("\t")
        u, l = (a, b) if q == "unlab" else (b, a)
        if float(e) < SEQ_E:
            rows.append((u, l, lab_fam[l], "blastn", float(e)))
print(f"BLASTN: {sum(r[3] == 'blastn' for r in rows)} hits", flush=True)

# nhmmer, both directions
for q, t in (("unlab", "lab"), ("lab", "unlab")):
    tbl = work / f"nhmmer_{q}_{t}.tbl"
    subprocess.run([_bin(HMMER_BIN, "nhmmer"), "--rna", "--cpu", CPU, "-E", str(SEQ_E), "--tblout", str(tbl),
                    "-o", _os.devnull, str(fa[q]), str(fa[t])], check=True)
    for line in open(tbl):
        if line.startswith("#"):
            continue
        f = line.split()
        tgt, qry, e = f[0], f[2], float(f[12])
        u, l = (qry, tgt) if q == "unlab" else (tgt, qry)
        if e < SEQ_E:
            rows.append((u, l, lab_fam[l], "nhmmer", e))
print(f"nhmmer: {sum(r[3] == 'nhmmer' for r in rows)} hits", flush=True)

# cmscan, unlabelled chains vs Rfam, relaxed
tbl = work / "cmscan.tbl"
subprocess.run([_bin(INFERNAL_BIN, "cmscan"), "--cpu", CPU, "-E", str(CMSCAN_E), "--fmt", "2",
                "--tblout", str(tbl), "-o", _os.devnull, RFAM_CM, str(fa["unlab"])], check=True)
for line in open(tbl):
    if line.startswith("#"):
        continue
    f = line.split()
    # fmt 2: idx target_name accession query_name ... E-value at column 17
    acc, qry, e = f[2], f[3], float(f[17])
    if e < CMSCAN_E:
        rows.append((qry, "", acc, "cmscan", e))
print(f"cmscan: {sum(r[3] == 'cmscan' for r in rows)} hits", flush=True)

assert all(r[0] in unlab_tags for r in rows)
with open(OUT, "w") as f:
    f.write("unlabelled_chain\tlabelled_chain\tfamily\tmethod\tevalue\n")
    for r in sorted(set(rows)):
        f.write("\t".join([r[0], r[1], r[2], r[3], f"{r[4]:.3g}"]) + "\n")
print(OUT)
