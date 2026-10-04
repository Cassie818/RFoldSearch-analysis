"""All searches for the sampled NuFold 9-14-2026 benchmark.

For both query sets (predicted, experimental) every method returns its complete
ranked output, so that neutral targets can be removed before evaluation:
    RFoldseek      candidate depth 200 / 500 / 2000 (whole pool ranked)
    RFoldseek-TM   candidate depth 200 / 500
    RiboSeek       candidate depth 200 / 500 (RS-20)
    BLASTN, nhmmer every hit at E < 10, best bit score per target
Every query is timed. Methods run one after another in a single process on an
otherwise idle machine; query preparation (tokens, RS-20 labels, sequences) is
done before timing starts.

Targets are identified by their row index in sample/accuracy_db.tsv.
Outputs: runs/<queryset>.json  {method: {qid: {"ranking": [...], "seconds": s}}}
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
import sys as _sys                                             # noqa: E402
_sys.path.insert(0, REPO)
import paths                                                   # noqa: E402
HERE = os.path.join(REPO, "files", "predicted_retrieval")      # results; reruns write db/, usalign/, runs/ here
ROOT = HERE                                                    # holds sample/
UP = paths.RFOLDSEEK_REPO
sys.path.insert(0, UP)
sys.path.insert(0, SCRIPT_DIR)
from build_databases import MAKEBLASTDB, TOOLS, path_of, seq_from_pdb   # noqa: E402

BLASTN = os.path.join(TOOLS, "blastn")
NHMMER = os.path.join(TOOLS, "hmmer", "bin", "nhmmer")
USALIGN = paths.USALIGN
NR80 = paths.NR80_DIR
DB = os.path.join(HERE, "db")
OUT = os.path.join(HERE, "runs")


def seq_from_cif(path):
    cols, rows, in_loop = {}, [], False
    for line in open(path):
        if line.startswith("_atom_site."):
            cols[line.strip().split(".", 1)[1]] = len(cols)
            in_loop = True
            continue
        if in_loop and line.startswith(("ATOM", "HETATM")):
            rows.append(line.split())
        elif in_loop and rows and line.startswith(("#", "loop_")):
            break
    ic, ia, im = cols["label_comp_id"], cols["label_asym_id"], cols["pdbx_PDB_model_num"]
    iseq, iauth, ins = cols["label_seq_id"], cols["auth_seq_id"], cols.get("pdbx_PDB_ins_code")
    model, chain = rows[0][im], rows[0][ia]
    out, last = [], None
    for r in rows:
        if r[im] != model or r[ia] != chain:
            continue
        key = (r[iseq], r[iauth], r[ins] if ins is not None else "")
        if key != last:
            last = key
            name = r[ic].strip('"')
            out.append(name if name in {"A", "C", "G", "U"} else "N")
    return "".join(out)


def load_queries(paper):
    manifest = list(csv.DictReader(open(os.path.join(ROOT, "sample", "accuracy_db.tsv")), delimiter="\t"))
    row_of = {r["urs_id"]: i for i, r in enumerate(manifest)}
    seqs = [l.strip() for l in open(os.path.join(DB, "sequences.fa")) if not l.startswith(">")]
    sets = {}
    p2p = json.load(open(os.path.join(ROOT, "sample", "p2p_queries.json")))["queries"]
    sets["p2p"] = [dict(id=q["urs_id"], index=row_of[q["urs_id"]], self_idx=row_of[q["urs_id"]],
                        path=path_of(q["urs_id"])) for q in p2p]
    exp = json.load(open(os.path.join(ROOT, "sample", "exp_queries.json")))["queries"]
    sets["exp"] = [dict(id=q["id"], paper_index=q["index"], self_idx=-1,
                        path=os.path.join(NR80, q["id"] + ".cif")) for q in exp]
    for q in sets["p2p"]:
        q["seq"] = seqs[q["index"]]
    for q in sets["exp"]:
        q["seq"] = seq_from_cif(q["path"])
        q["entry"] = paper[q["paper_index"]]
    return manifest, sets


def run_token(sets, manifest):
    from rfoldseek.search import Searcher
    t0 = time.perf_counter()
    s = Searcher(os.path.join(DB, "rfoldseek.pt"), evalue_params_path=os.path.join(DB, "__none__.json"))
    print(f"rfoldseek load+index {time.perf_counter()-t0:.1f}s", flush=True)
    assert [e["id"] for e in s.ents] == [path_of(r["urs_id"]) for r in manifest], "db order != manifest"
    ti = {cid: i for i, cid in enumerate(s.ids)}
    for q in sets.get("p2p", []):          # predicted queries live in the database
        q["entry"] = s.ents[q["index"]]
    first = next(iter(sets.values()))[0]    # warm up numba on any query
    s.search(np.asarray(first["entry"]["tokens"], dtype=np.int64), top_k=5)
    res = {k: {} for k in sets}
    for name, top_n, rerank in (("RFoldseek n500", 500, "atm"), ("RFoldseek-TM n200", 200, "tm"),
                                ("RFoldseek-TM n500", 500, "tm")):
        for qs, queries in sets.items():
            out = {}
            for q in queries:
                tok = np.asarray(q["entry"]["tokens"], dtype=np.int64)
                kw = dict(rerank=rerank, query_entry=q["entry"])
                if rerank == "tm":
                    kw["usalign"] = USALIGN
                t = time.perf_counter()
                hits = s.search(tok, top_k=top_n, top_n=top_n, self_idx=q["self_idx"], **kw)
                out[q["id"]] = dict(seconds=time.perf_counter() - t, ranking=[ti[h["target"]] for h in hits])
            res[qs][name] = out
            print(f"  {qs} {name}: median {np.median([v['seconds'] for v in out.values()])*1000:.1f} ms", flush=True)
    return res


def run_riboseek(sets, res):
    from riboseek import Searcher as RS
    meta = json.load(open(os.path.join(DB, "riboseek_sa20.json.meta.json")))
    k2i = meta["key_to_index"]
    i2k = {v: k for k, v in k2i.items()}
    rs = RS.from_pretrained("sa20", db=os.path.join(DB, "riboseek_sa20.json"))
    rs._ensure_prefilter()
    labels = {}
    for q in sets["exp"]:
        labels[q["id"]] = rs.encode(q["path"])
    for q in sets["p2p"]:
        labels[q["id"]] = i2k[q["index"]]           # key: RiboSeek excludes the query itself
    rs.search(labels[sets["exp"][0]["id"]], top_n=5)
    for cand in (200, 500):
        for qs, queries in sets.items():
            out = {}
            for q in queries:
                t = time.perf_counter()
                hits = rs.search(labels[q["id"]], top_n=cand, prefilter=True, prefilter_candidates=cand)
                out[q["id"]] = dict(seconds=time.perf_counter() - t, ranking=[k2i[h["chain"]] for h in hits])
            res[qs][f"RiboSeek c{cand}"] = out
            print(f"  {qs} RiboSeek c{cand}: median {np.median([v['seconds'] for v in out.values()])*1000:.1f} ms", flush=True)


def run_sequence(sets, res):
    fa = os.path.join(DB, "sequences.fa")
    with tempfile.TemporaryDirectory(prefix="seqsearch_") as tmp:
        for qs, queries in sets.items():
            for tool in ("BLASTN", "nhmmer"):
                res[qs][tool] = {}
            for q in queries:
                qfa = os.path.join(tmp, "q.fa")
                open(qfa, "w").write(f">q\n{q['seq']}\n")
                t = time.perf_counter()
                p = subprocess.run([BLASTN, "-task", "blastn", "-query", qfa, "-db", fa, "-strand", "plus",
                                    "-evalue", "10", "-num_threads", "1", "-max_target_seqs", "100000",
                                    "-outfmt", "6 sseqid bitscore evalue"],
                                   capture_output=True, text=True, check=True)
                dt = time.perf_counter() - t
                best = {}
                for line in p.stdout.splitlines():
                    sid, bits, ev = line.split("\t")
                    i = int(sid[1:])
                    if i not in best or float(bits) > best[i][0]:
                        best[i] = (float(bits), float(ev))
                order = sorted((i for i in best if i != q["self_idx"]), key=lambda i: (-best[i][0], i))
                res[qs]["BLASTN"][q["id"]] = dict(seconds=dt, ranking=order, evalue=[best[i][1] for i in order])
                tbl = os.path.join(tmp, "o.tbl")
                t = time.perf_counter()
                subprocess.run([NHMMER, "--cpu", "1", "--watson", "--rna", "-E", "10", "--noali", "--tblout", tbl,
                                "-o", os.devnull, qfa, fa], capture_output=True, text=True, check=True)
                dt = time.perf_counter() - t
                best = {}
                for line in open(tbl):
                    if not line.startswith("#"):
                        f = line.split()
                        i, ev, bits = int(f[0][1:]), float(f[12]), float(f[13])
                        if i not in best or bits > best[i][0]:
                            best[i] = (bits, ev)
                order = sorted((i for i in best if i != q["self_idx"]), key=lambda i: (-best[i][0], i))
                res[qs]["nhmmer"][q["id"]] = dict(seconds=dt, ranking=order, evalue=[best[i][1] for i in order])
            for tool in ("BLASTN", "nhmmer"):
                sec = [v["seconds"] for v in res[qs][tool].values()]
                print(f"  {qs} {tool}: median {np.median(sec)*1000:.1f} ms", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    paper = torch.load(os.path.join(REPO, "files", "models", "db.pt"),
                       map_location="cpu", weights_only=False)["entries"]
    manifest, sets = load_queries(paper)
    print(f"queries: p2p {len(sets['p2p'])}, exp {len(sets['exp'])}", flush=True)
    res = run_token(sets, manifest)
    run_riboseek(sets, res)
    run_sequence(sets, res)
    for qs in res:
        json.dump(res[qs], open(os.path.join(OUT, f"{qs}.json"), "w"))
    print("DONE searches", flush=True)


if __name__ == "__main__":
    main()
