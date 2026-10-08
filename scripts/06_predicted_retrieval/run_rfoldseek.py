"""RFoldseek on the Fig. 5 benchmark at the reported candidate depth (500)"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_searches as RS

DEPTHS = (500,)


def main():
    import torch
    from rfoldseek.search import Searcher
    paper = torch.load(os.path.join(RS.REPO, "files", "models", "db.pt"), map_location="cpu",
                       weights_only=False)["entries"]
    _, sets = RS.load_queries(paper)
    unl = json.load(open(os.path.join(RS.ROOT, "sample", "exp_unlabelled_queries.json")))
    sets["exp_unlabelled"] = [dict(id=q["id"], self_idx=-1, entry=paper[q["index"]]) for q in unl]
    s = Searcher(os.path.join(RS.DB, "rfoldseek.pt"), evalue_params_path=os.path.join(RS.DB, "__none__.json"))
    ti = {cid: i for i, cid in enumerate(s.ids)}
    for q in sets.get("p2p", []):
        q["entry"] = s.ents[q["index"]]
    first = next(iter(sets.values()))[0]
    s.search(np.asarray(first["entry"]["tokens"], dtype=np.int64), top_k=5, query_entry=first["entry"],
             self_idx=first["self_idx"])
    for qs, queries in sets.items():
        path = os.path.join(RS.OUT, f"{qs}.json")
        runs = json.load(open(path))
        for depth in DEPTHS:
            out = {}
            for q in queries:
                tok = np.asarray(q["entry"]["tokens"], dtype=np.int64)
                t = time.perf_counter()
                hits = s.search(tok, top_k=depth, top_n=depth, self_idx=q["self_idx"], query_entry=q["entry"])
                out[q["id"]] = dict(seconds=time.perf_counter() - t, ranking=[ti[h["target"]] for h in hits])
            runs[f"RFoldseek n{depth}"] = out
            print(f"{qs} n{depth}: {len(out)} queries, median "
                  f"{np.median([v['seconds'] for v in out.values()]) * 1000:.1f} ms", flush=True)
        json.dump(runs, open(path, "w"))


if __name__ == "__main__":
    main()
