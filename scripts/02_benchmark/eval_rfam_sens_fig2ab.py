"""Assemble a TEST-ONLY per_query_sens.json for Fig 2:
per-query family & clan Sens@1FP for NEW RFoldseek (production_new deployed
pipeline) + real RiboSeek (sa20) + US-align/RMalign/BLASTN (subset from the
existing all-query per_query_sens.json). Same 106 fam / 79 clan test query set."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH
_sys.path.insert(0, SRC_SEARCH); _sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path
import os,sys,json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from collections import defaultdict
import numpy as np, torch

from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels
from smith_waterman import smith_waterman_numba
K=80; TOPN=100; KMER_K=4; THRESH_FRAC=0.75; GO=-2.0; GE=-0.5
ND="files/production_new"
db=torch.load(f"{ND}/db.pt",map_location="cpu",weights_only=False); ents=db["entries"]; N=len(ents); id2i={str(e["id"]):i for i,e in enumerate(ents)}
fams=attach_labels(ents,load_labels("Splits/labels.tsv")); clans=attach_labels(ents,load_labels("Splits/labels.tsv",label_column="clan_acc"))
fam_pool=np.array(fams,dtype=object); clan_pool=np.array(clans,dtype=object)
# existing all-query per-query file provides the canonical query id ordering
pq=json.load(open("files/production/retrieval/per_query_sens.json"))
fam_qids=pq["_query_ids"]["family"]; clan_qids=pq["_query_ids"]["clan"]
def idl(p): return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]
test_dis=set(id2i[c] for c in idl("Splits/pdb_out/test.json") if c in id2i)
fam_test_idx=[i for i,q in enumerate(fam_qids) if q in test_dis]   # positions within the 468 arrays
clan_test_idx=[i for i,q in enumerate(clan_qids) if q in test_dis]
fam_test_q=[fam_qids[i] for i in fam_test_idx]                     # db indices of the 106 test fam queries
clan_test_q=[clan_qids[i] for i in clan_test_idx]
print(f"test fam queries={len(fam_test_q)}  clan queries={len(clan_test_q)}",flush=True)
# ---- NEW RFoldseek deployed pipeline rankings ----
toks=[np.asarray(e["tokens"],dtype=np.int64) for e in ents]
sm=torch.load(f"{ND}/sub_matrix.pt",map_location="cpu",weights_only=False); sub=np.asarray(sm["matrix"].numpy(),dtype=np.float32)
ck=torch.load(f"{ND}/desc_vqvae_best.pt",map_location="cpu",weights_only=False)
cb=ck["model"]["codebook.embeddings"].numpy(); cb=cb/(np.linalg.norm(cb,axis=1,keepdims=True)+1e-9); cos=(cb@cb.T).astype(np.float32)
powers=np.array([K**(KMER_K-1-i) for i in range(KMER_K)],dtype=np.int64)
idx=defaultdict(list)
for ti,tt in enumerate(toks):
    for p in range(len(tt)-KMER_K+1): idx[int(np.dot(tt[p:p+KMER_K],powers))].append((ti,p))
smith_waterman_numba(np.zeros(4,np.int64),np.zeros(4,np.int64),sub,GO,GE)
thr=THRESH_FRAC*KMER_K
def expand(qk):
    out={int(np.dot(qk,powers))}
    for pos in range(KMER_K):
        for tok in range(K):
            if tok!=qk[pos] and (KMER_K-1)*1.0+cos[qk[pos],tok]>=thr:
                tr=qk.copy(); tr[pos]=tok; out.add(int(np.dot(tr,powers)))
    return list(out)
def rank_new(q):
    tt=toks[q]; td=defaultdict(dict)
    for qp in range(len(tt)-KMER_K+1):
        for kid in expand(tt[qp:qp+KMER_K]):
            for ti,tp in idx.get(kid,()):
                if ti!=q: d=qp-tp; td[ti][d]=td[ti].get(d,0)+1
    sc=[(ti,max((c for c in dm.values() if c>=2),default=-1)) for ti,dm in td.items()]; sc=[x for x in sc if x[1]>0]; sc.sort(key=lambda x:-x[1])
    cand=np.array([x[0] for x in sc[:TOPN]],np.int64)
    if len(cand)==0: return np.zeros(0,np.int64)
    return cand[np.argsort(-np.array([smith_waterman_numba(tt,toks[t],sub,GO,GE) for t in cand]),kind="stable")]
# ---- real RiboSeek rankings from cached encodings ----
from riboseek import Searcher
import importlib.resources as R
enc=json.load(open("files/ablation/riboseek_real_encoded.json"))
with R.as_file(R.files("riboseek.data").joinpath("demo_db.json")) as p: demo=str(p)
sr0=Searcher.from_pretrained("sa20", db=demo)
srr=Searcher(sr0.alphabet, enc, gap_penalty=-2.0, prefilter_k=6)
def rank_ribo(q):
    res=srr.search(str(q), top_n=100, prefilter=True, prefilter_candidates=100)
    return np.asarray([int(h["chain"]) for h in res if int(h["chain"])!=q],dtype=np.int64)
def per_query_sens(rank_fn):
    fam_sens={}; clan_sens={}
    va_all=np.arange(N)
    allq=set(fam_test_q)|set(clan_test_q)
    ranks={}
    for q in sorted(allq): ranks[q]=rank_fn(q)
    for q in fam_test_q:
        r=ranks[q]; va=va_all!=q
        rel=(fam_pool==fam_pool[q])&va
        neu=((clan_pool==clan_pool[q])&(fam_pool!=fam_pool[q])&va) if clan_pool[q] else None
        fam_sens[q]=first_fp_sensitivity(r,rel,neu) if len(r) else 0.0
    for q in clan_test_q:
        r=ranks[q]; va=va_all!=q
        rel=(clan_pool==clan_pool[q])&(fam_pool!=fam_pool[q])&va
        neu=(fam_pool==fam_pool[q])&va
        clan_sens[q]=first_fp_sensitivity(r,rel,neu) if len(r) else 0.0
    return fam_sens,clan_sens
print("computing NEW RFoldseek per-query...",flush=True)
nf,nc=per_query_sens(rank_new)
print("computing RiboSeek per-query...",flush=True)
rf,rc=per_query_sens(rank_ribo)
# assemble output in the same array order as the test subset positions
out={"_query_ids":{"family":fam_test_q,"clan":clan_test_q}}
def pack(fam_map,clan_map):
    fs=[fam_map[q] for q in fam_test_q]; cs=[clan_map[q] for q in clan_test_q]
    return {"family":{"sens":fs,"n":len(fs),"mean":float(np.mean(fs))},
            "clan":{"sens":cs,"n":len(cs),"mean":float(np.mean(cs))}}
out["RFoldseek"]=pack(nf,nc)
out["RiboSeek"]=pack(rf,rc)
# subset existing tools to test positions
for t in ["US-align","RMalign","BLASTN"]:
    fs=[pq[t]["family"]["sens"][i] for i in fam_test_idx]
    cs=[pq[t]["clan"]["sens"][i] for i in clan_test_idx]
    out[t]={"family":{"sens":fs,"n":len(fs),"mean":float(np.mean(fs))},
            "clan":{"sens":cs,"n":len(cs),"mean":float(np.mean(cs))}}
json.dump(out,open("files/production/retrieval/per_query_sens_test.json","w"),indent=2)
print("\n=== TEST-ONLY per-query means ===",flush=True)
for t in ["RFoldseek","US-align","RMalign","RiboSeek","BLASTN"]:
    print(f"  {t:12s} famSens={out[t]['family']['mean']:.3f}  clanSens={out[t]['clan']['mean']:.3f}",flush=True)
print("\nwrote files/production/retrieval/per_query_sens_test.json\nDONE",flush=True)
