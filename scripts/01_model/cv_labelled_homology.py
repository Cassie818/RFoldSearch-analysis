"""Sequence-level homology among labelled CV-pool chains of different Rfam families."""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, BLAST_BIN, SRC_SEARCH, resolve

_sys.path.insert(0, SRC_SEARCH)
SEQS = resolve("files/production/chain_sequences.tsv")
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
OUT = _os.path.join(_REPO, "files/ablation/cv_labelled_seqhits.tsv")
_os.chdir(DATA_ROOT)
import csv, json, re, subprocess, tempfile
from pathlib import Path
import torch

from prefilter_rfam_benchmark import attach_labels, load_labels

HMMER_BIN = _os.environ.get(
    "RFOLDSEARCH_HMMER_BIN", _os.environ.get("RFOLDSEEK_HMMER_BIN", ""))
SEQ_E = 0.01
CPU = str(_os.cpu_count() or 4)


def _bin(d, name):
    return _os.path.join(d, name) if d else name


ents = torch.load("files/production/db.pt", map_location="cpu", weights_only=False)["entries"]
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
fams = attach_labels(ents, load_labels("Splits/labels.tsv"))


def idl(p):
    return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]


pool = [id2i[c] for c in idl("Splits/pdb_out/train.json") + idl("Splits/pdb_out/val.json") if c in id2i]
lab = [d for d in pool if fams[d]]
tag = {d: _os.path.basename(str(ents[d]["id"])).removesuffix(".cif") for d in lab}
fam_of = {tag[d]: fams[d] for d in lab}
seqs = {r["chain_id"]: r["sequence"] for r in csv.DictReader(open(SEQS), delimiter="\t")}
work = Path(tempfile.mkdtemp(prefix="cv_labhom_"))
fa = work / "lab.fa"
with open(fa, "w") as f:
    for d in lab:
        f.write(f">{tag[d]}\n{re.sub(r'[^ACGU]', 'N', seqs[tag[d]].upper().replace('T', 'U'))}\n")
rows = set()

subprocess.run([_bin(BLAST_BIN, "makeblastdb"), "-in", str(fa), "-dbtype", "nucl", "-out", str(work / "lab")],
               check=True, capture_output=True)
out = work / "blast.tsv"
subprocess.run([_bin(BLAST_BIN, "blastn"), "-task", "blastn", "-query", str(fa), "-db", str(work / "lab"),
                "-evalue", str(SEQ_E), "-dust", "no", "-num_threads", CPU, "-max_target_seqs", "1000",
                "-outfmt", "6 qseqid sseqid evalue bitscore", "-out", str(out)], check=True, capture_output=True)
for line in open(out):
    a, b, e, _ = line.rstrip("\n").split("\t")
    if float(e) < SEQ_E and fam_of[a] != fam_of[b]:
        rows.add((min(a, b), max(a, b), "blastn", float(e)))
print(f"BLASTN: {sum(r[2] == 'blastn' for r in rows)} cross-family hits", flush=True)

tbl = work / "nhmmer.tbl"
subprocess.run([_bin(HMMER_BIN, "nhmmer"), "--rna", "--cpu", CPU, "-E", str(SEQ_E), "--tblout", str(tbl),
                "-o", _os.devnull, str(fa), str(fa)], check=True)
for line in open(tbl):
    if line.startswith("#"):
        continue
    f = line.split()
    a, b, e = f[0], f[2], float(f[12])
    if e < SEQ_E and fam_of[a] != fam_of[b]:
        rows.add((min(a, b), max(a, b), "nhmmer", e))
print(f"nhmmer: {sum(r[2] == 'nhmmer' for r in rows)} cross-family hits", flush=True)

with open(OUT, "w") as f:
    f.write("chain_a\tchain_b\tmethod\tevalue\n")
    for r in sorted(rows):
        f.write("\t".join([r[0], r[1], r[2], f"{r[3]:.3g}"]) + "\n")
print(OUT)
