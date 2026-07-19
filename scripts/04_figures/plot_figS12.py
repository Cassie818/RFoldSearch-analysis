"""Fig S12: similar-k-mer scoring ablation for the prefilter, 5-fold PDB-disjoint
CV, K=80 deployed VQ-VAE + Structure-BLOSUM reranking. Same style as Fig S7."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH
_sys.path.insert(0, SRC_SEARCH); _sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"]="sans-serif"
matplotlib.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]
import matplotlib.pyplot as plt
d=json.load(open(Path(DATA_ROOT)/"files/ablation/cv_prefilter_score_5fold.json"))
NAMES=[("E4","no expansion\n(exact only)"),
       ("blosum>=.25","Structure-BLOSUM\n$S\\geq0.25$"),
       ("blosum>=0","Structure-BLOSUM\n$S\\geq0$"),
       ("cos>=0","codebook cosine\n$\\cos\\geq0$")]
EXPANSION={"E4":1,"blosum>=.25":9,"blosum>=0":66,"cos>=0":161}
xs=np.arange(len(NAMES))
fam_m=np.array([d[k[0]]["fam_sens_mean"] for k in NAMES]); fam_s=np.array([d[k[0]]["fam_sens_std"] for k in NAMES])
clan_m=np.array([d[k[0]]["clan_sens_mean"] for k in NAMES]); clan_s=np.array([d[k[0]]["clan_sens_std"] for k in NAMES])
fig,ax=plt.subplots(figsize=(8,4.8))
C1="#5B7C99"; C2="#B08D5C"
for m,s,c,mk,lab in ((fam_m,fam_s,C1,"o","Family Sens@1FP"),(clan_m,clan_s,C2,"s","Clan Sens@1FP")):
    ax.plot(xs,m,mk+"-",color=c,linewidth=2,markersize=7,label=lab,
            markerfacecolor=c,markeredgecolor="white",markeredgewidth=1,zorder=3)
    for x,v,e in zip(xs,m,s):
        ax.vlines(x,v,v+e,colors="#333333",linewidth=1.0,alpha=0.9,zorder=2)
        ax.hlines(v+e,x-0.09,x+0.09,colors="#333333",linewidth=1.0,alpha=0.9,zorder=2)
    for x,v,e in zip(xs,m,s): ax.text(x,v+e+0.010,f"{v:.3f}",ha="center",fontsize=8,color=c)
# expansion factor as a secondary annotation row
for x,(k,_) in zip(xs,NAMES):
    ax.text(x,-0.175,f"{EXPANSION[k]}$\\times$",ha="center",va="top",fontsize=9,
            color="#666666",transform=ax.get_xaxis_transform())
ax.text(-0.62,-0.175,"$k$-mers looked up",ha="left",va="top",fontsize=8.5,color="#666666",
        transform=ax.get_xaxis_transform(),style="italic")
ax.set_xlabel("Similar-$k$-mer scoring criterion",fontsize=12)
ax.xaxis.set_label_coords(0.5,-0.29)
ax.set_ylabel("Sens@1FP",fontsize=12)
ax.set_xticks(xs); ax.set_xticklabels([n[1] for n in NAMES],fontsize=9.5)
ax.set_xlim(-0.65,len(NAMES)-0.35); ax.set_ylim(0,0.52)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis="both",labelsize=10)
ax.legend(loc="center left",frameon=False,fontsize=11,ncol=1,bbox_to_anchor=(0.03,0.42),
          handlelength=2.0)
fig.tight_layout(rect=(0,0.06,1,1))
OUT=str(Path(FIG_DIR)/"S12.png")
fig.savefig(OUT,dpi=200,bbox_inches="tight"); print(f"wrote {OUT}")
