"""Fig 5: swing (acquiescence) vs net bias are orthogonal axes.
x = framing swing (half), y = net polarity-aligned bias. Colour by origin.
Mistral and Qwen share an x but differ wildly in y -> the validity proof."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROV_COLOR = {"EU":"#E8A33D","CN":"#C0392B","US":"#2E6FB5"}
d = pd.read_csv("bias_decomposition.csv")
fig, ax = plt.subplots(figsize=(7.4,5.6))
ax.axhline(0, color="#999", lw=.9, ls="--")
for _,r in d.iterrows():
    c = PROV_COLOR[r.origin]
    ax.scatter(r.swing_half, r.NET_bias, s=150, color=c, edgecolor="k", lw=.6, zorder=3)
    ax.annotate(r.model, (r.swing_half, r.NET_bias), fontsize=7.5, fontweight="bold",
                xytext=(6,3), textcoords="offset points", color=c)
# highlight the Mistral vs Qwen contrast (same swing, opposite bias)
mi = d[d.model=="mistral-small-2603"].iloc[0]; qw = d[d.model=="qwen3.6-plus"].iloc[0]
ax.annotate("", xy=(qw.swing_half, qw.NET_bias), xytext=(mi.swing_half, mi.NET_bias),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2, ls=":"))
ax.text(0.225, -0.33, "same swing,\nopposite bias", fontsize=8, style="italic", color="#555", ha="center")
ax.set_xlabel("Framing swing (half-difference between framings)  =  acquiescence magnitude", fontsize=9.5)
ax.set_ylabel("Net polarity-aligned bias\n(← Pro-China      Pro-U.S. →)", fontsize=9.5)
ax.set_title("Acquiescence and bias are orthogonal:\nswinging is not the same as being biased", fontsize=11, fontweight="bold")
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=PROV_COLOR[o],
          markersize=10,label=o) for o in ["EU","CN","US"]], loc="lower left", frameon=True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("fig5_swing_vs_bias.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote fig5_swing_vs_bias.png")
