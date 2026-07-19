"""Regenerate Fig S2 with cleaner error bars + shared bottom legend."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH
_sys.path.insert(0, SRC_SEARCH); _sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["font.family"]="sans-serif"
matplotlib.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]
import matplotlib.pyplot as plt
d=json.load(open("files/ablation/cv_s2_clan_5fold.json"))
BLOCKS=["g_nn1","g_nn2","g_bp","g_st","g_ctx"]
LABELS=["$g_{\\mathrm{nn}}^{(1)}$","$g_{\\mathrm{nn}}^{(2)}$","$g_{\\mathrm{bp}} + m_{\\mathrm{bp}}$","$g_{\\mathrm{st}} + m_{\\mathrm{st}}$","$g_{\\mathrm{ctx}}$"]
KS=[20,40,60,80]
COLORS=["#5B7C99","#B08D5C","#7DA087","#9B6B8A"]
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
for pi,(mkey,skey,ylab,ylim) in enumerate([("fam_sens_mean","fam_sens_std","Family Sens@1FP",0.55),("clan_sens_mean","clan_sens_std","Clan Sens@1FP",0.12)]):
    ax=axes[pi]
    x=np.arange(len(BLOCKS)); w=0.21
    for ki,K in enumerate(KS):
        vals=np.array([d[f"{b}|K{K}"][mkey] for b in BLOCKS])
        errs=np.array([d[f"{b}|K{K}"][skey] for b in BLOCKS])
        yerr_asym=np.vstack([np.zeros_like(errs),errs])  # lower=0, upper=std
        bar_xs=x+(ki-1.5)*w
        ax.bar(bar_xs,vals,w,yerr=yerr_asym,color=COLORS[ki],
               label=f"K = {K}",edgecolor="none",
               error_kw=dict(elinewidth=1.0,ecolor="#333333",capsize=0,alpha=0.9))
        # 只在顶部画cap短横线
        cap_half=w*0.28
        for xb,v,e in zip(bar_xs,vals,errs):
            ax.hlines(v+e,xb-cap_half,xb+cap_half,colors="#333333",linewidth=1.0,alpha=0.9)
        for j,v in enumerate(vals): ax.text(x[j]+(ki-1.5)*w,v+errs[j]+ylim*0.022,f"{v:.3f}",ha="center",fontsize=5.5,color=COLORS[ki])
    ax.set_xticks(x); ax.set_xticklabels(LABELS,fontsize=11)
    ax.set_ylabel(ylab,fontsize=11); ax.set_ylim(0,ylim)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(-0.09,1.04,"A" if pi==0 else "B",transform=ax.transAxes,fontsize=16,fontweight="bold",ha="right")
# 底部统一legend
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,fontsize=12,bbox_to_anchor=(0.5,-0.02),handlelength=1.4,columnspacing=2.2)
fig.tight_layout(rect=(0,0.05,1,1))
OUT = str(Path(FIG_DIR) / "S2.png")
fig.savefig(OUT,dpi=200,bbox_inches="tight")
print(f"wrote {OUT}")
