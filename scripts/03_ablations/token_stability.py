"""Is the VQ-VAE alphabet more stable than k-means at large codebook size?

Two alignment-free probes on the identical 17-D descriptors:

  (1) Assignment stability -- perturb each descriptor by Gaussian noise of a
      given fraction of its (unit) standard deviation, re-tokenise, and record
      how often the token changes. A quantiser whose cell boundaries cut through
      dense regions flips more often.
  (2) Codebook usage -- the entropy of the token distribution, reported as the
      effective number of states exp(H). A codebook that is nominally K states
      but uses far fewer is not really a finer alphabet.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch
from sklearn.cluster import MiniBatchKMeans

PC = os.path.expanduser("~/Desktop/RFoldseek-paper-code")
sys.path.insert(0, os.path.join(PC, "scripts", "03_ablations"))
sys.path.insert(0, os.path.join(PC, "src", "models"))
os.environ.setdefault("RFS_ALPHABET_CONTROL_MATRIX", "native")
import fig2ef as F  # noqa: E402
from vqvae import DescriptorVQVAE  # noqa: E402

SPLITS = os.path.expanduser("~/Desktop/RFoldSeek/Splits/pdb_out")
KEEP = sorted(list(range(0, 10)) + list(range(42, 49)))
NOISE = [0.05, 0.10, 0.20]
KS = [10, 20, 40, 80]
SEED = 0


def entropy_states(tok, K):
    c = np.bincount(tok, minlength=K).astype(float)
    p = c[c > 0] / c.sum()
    return float(np.exp(-(p * np.log(p)).sum())), int((c == 0).sum())


def main():
    db = torch.load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
    ents = db["entries"]
    id2i = {os.path.basename(str(e["id"])): i for i, e in enumerate(ents)}
    raw, split_of = {}, {}
    for sp in ("train", "val", "test"):
        D = torch.load(f"{SPLITS}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
        off = 0
        for line in open(f"{SPLITS}/{sp}.json"):
            line = line.strip()
            if not line:
                continue
            di = id2i[os.path.basename(json.loads(line)["id"])]
            L = len(ents[di]["tokens"])
            raw[di] = D[off:off + L].astype(np.float32)
            split_of[di] = sp
            off += L
    tr = np.concatenate([raw[i][:, KEEP] for i in raw if split_of[i] == "train"])
    mn, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-6] = 1.0
    X = np.concatenate([raw[i][:, KEEP] for i in sorted(raw)])          # all 885 chains
    Xn = ((X - mn) / sd).astype(np.float64)
    Xtr = ((np.concatenate([raw[i][:, KEEP] for i in raw if split_of[i] == "train"]) - mn) / sd).astype(np.float64)
    print(f"residues: {len(Xn)}  (train {len(Xtr)})", flush=True)

    # deployed VQ-VAE: encode the canonical 49-D layout with the inactive dims zeroed
    ck = torch.load(F.resolve("files/models/desc_vqvae_best.pt"), map_location="cpu", weights_only=False)
    m = DescriptorVQVAE(d_descriptor=ck["d_descriptor"], n_codes=ck["n_codes"],
                        d_hidden=ck["d_hidden"], d_latent=ck["d_latent"],
                        n_enc_layers=ck["n_enc_layers"], n_dec_layers=ck["n_dec_layers"])
    m.load_state_dict(ck["model"]); m.eval()
    raw_all = np.concatenate([raw[i] for i in sorted(raw)])
    full_mn = np.zeros(49, np.float32); full_sd = np.ones(49, np.float32)
    full_mn[KEEP], full_sd[KEEP] = mn, sd

    def vq_tokens(mat49):
        z = ((mat49 - full_mn) / full_sd).astype(np.float32)
        z[:, [i for i in range(49) if i not in KEEP]] = 0.0
        with torch.no_grad():
            out = m(torch.from_numpy(z))
        idx = out[2] if isinstance(out, (tuple, list)) and len(out) > 2 else out["indices"]
        return np.asarray(idx).reshape(-1)

    rng = np.random.default_rng(SEED)
    base = {"VQ-VAE_K80": vq_tokens(raw_all)}
    kms = {}
    for K in KS:
        km = MiniBatchKMeans(n_clusters=K, batch_size=10000, max_iter=200,
                             random_state=42, n_init=3).fit(Xtr)
        kms[K] = km
        base[f"KMeans_K{K}"] = km.predict(Xn)
    print("baseline tokenisations done", flush=True)

    res = {"_config": {"noise_levels": NOISE, "seed": SEED, "n_residues": int(len(Xn))}}
    print("\n%-14s %6s %8s %s" % ("quantiser", "K", "eff.K", "  ".join(f"flip@{n}" for n in NOISE)))
    for name, tok in base.items():
        K = 80 if name.startswith("VQ") else int(name.split("_K")[1])
        eff, dead = entropy_states(tok, K)
        flips = []
        for nz in NOISE:
            if name.startswith("VQ"):
                pert = raw_all + rng.normal(0, nz, raw_all.shape).astype(np.float32) * full_sd
                t2 = vq_tokens(pert)
            else:
                t2 = kms[K].predict(Xn + rng.normal(0, nz, Xn.shape))
            flips.append(float((t2 != tok).mean()))
        res[name] = {"K": K, "effective_states": eff, "dead_states": dead,
                     "flip_rate": dict(zip(map(str, NOISE), flips))}
        print("%-14s %6d %8.1f %s" % (name, K, eff, "  ".join(f"{f:7.3f}" for f in flips)), flush=True)
    json.dump(res, open(f"{PC}/files/ablation/token_stability.json", "w"), indent=2)
    print(f"\nWrote {PC}/files/ablation/token_stability.json")


if __name__ == "__main__":
    main()
