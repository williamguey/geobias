# -*- coding: utf-8 -*-
"""Supplementary transfer figure: the instrument applied unchanged to two
non-geopolitical domains. Panels A,B: swing vs net per model. Panel C: per-model
swing is partially stable across the geopolitical and new domains."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

PROV = {"mistralai/mistral-small-2603":"EU","deepseek/deepseek-v4-flash":"CN",
        "bytedance-seed/seed-2.0-lite":"CN","qwen/qwen3.6-plus":"CN",
        "minimax/minimax-m2.7":"CN","z-ai/glm-5.1":"CN","openai/gpt-5.3-chat":"US",
        "openai/gpt-4o-mini":"US","anthropic/claude-sonnet-4.6":"US",
        "google/gemini-3.1-flash-lite":"US","x-ai/grok-4.3":"US"}
COL = {"EU":"#E8A33D","CN":"#C0392B","US":"#2E6FB5"}
df = pd.read_excel("pilot_transfer.xlsx")
df["M"] = df.Model.map(lambda m: m.split("/")[-1])
df["O"] = df.Model.map(PROV)

def ns(s):
    a = s[s.Framing=="Affirmative"]["aligned"].mean(); r = s[s.Framing=="Reverse"]["aligned"].mean()
    return (a+r)/2, abs(a-r)/2

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, dom, ttl in [(axes[0],"Cultural values","a  Cultural values\n(+ tradition / collectivism, - autonomy / individualism)"),
                     (axes[1],"Scientific consensus","b  Scientific consensus\n(+ consensus, - contrarian)")]:
    for m in df.M.unique():
        s = df[(df.M==m)&(df.Domain==dom)]; net, sw = ns(s); o = s.O.iloc[0]
        ax.scatter(sw, net, s=130, color=COL[o], edgecolor="k", lw=.6, zorder=3)
        if m=="mistral-small-2603":
            ax.annotate("Mistral (acquiescer)", (sw,net), fontsize=8, fontweight="bold",
                        xytext=(8,0), textcoords="offset points", color=COL["EU"])
    ax.axhline(0, color="#999", lw=.8, ls="--")
    ax.set_xlabel("swing (acquiescence)", fontsize=9.5)
    ax.set_ylabel("net bias", fontsize=9.5); ax.set_title(ttl, fontsize=9.5, fontweight="bold", loc="left")
    ax.set_ylim(-2.2, 2.2)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)

# panel C: cross-domain swing stability
geo = pd.read_csv("bias_decomposition.csv"); gsw = dict(zip(geo.model, geo.swing_half))
pts = []
for m in df.M.unique():
    _, sw = ns(df[df.M==m]); pts.append((m, gsw.get(m), sw, PROV.get("/".join([k for k in PROV if k.split("/")[-1]==m]) or "", "US")))
C = pd.DataFrame(pts, columns=["m","geo","new","o"]).dropna()
axС = axes[2]
for _,r in C.iterrows():
    axС.scatter(r.geo, r["new"], s=130, color=COL[r.o], edgecolor="k", lw=.6, zorder=3)
axС.annotate("Mistral", (C[C.m=="mistral-small-2603"].geo.iloc[0], C[C.m=="mistral-small-2603"]["new"].iloc[0]),
             fontsize=8, fontweight="bold", xytext=(6,2), textcoords="offset points", color=COL["EU"])
rho = stats.spearmanr(C.geo, C["new"]); pear = stats.pearsonr(C.geo, C["new"])
axС.set_xlabel("swing in geopolitics", fontsize=9.5); axС.set_ylabel("swing in new domains", fontsize=9.5)
axС.set_title(f"c  Acquiescence is partially stable\nSpearman rho = {rho[0]:.2f}, Pearson r = {pear[0]:.2f} (n=11)",
              fontsize=9.5, fontweight="bold", loc="left")
for sp in ("top","right"): axС.spines[sp].set_visible(False)
fig.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=COL[o],markersize=9,label=o) for o in ["EU","CN","US"]],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5,-0.04))
fig.suptitle("The forced-choice instrument transfers to non-geopolitical domains", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0,0.02,1,0.96]); fig.savefig("fig_S_transfer.png", dpi=190, bbox_inches="tight"); plt.close(fig)
print("wrote fig_S_transfer.png")
