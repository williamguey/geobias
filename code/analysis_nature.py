"""Nature-Comms analyses from the 64-iter dataset (no API). Produces:
  variance_decomposition.csv        (#1 ANOVA eta-squared + incremental R^2)
  language_gating.csv + fig_language_gating.png   (#2 EN->ZH shift)
  pca_coordinates.csv + fig_ideological_map.png   (#5 ideological map)
  topic_origin_effect.csv           (#6 convergence topics)
  acquiescence.csv                  (#3 framing yea-saying)
  ANALYSIS_SUMMARY.md               (headline numbers)
All scores polarity-aligned: + = Pro-US, - = Pro-China."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import geobias_app as g

def align(s, t, f):
    return -s if ((g.AFFIRMATIVE_POLE[t] == "CN") == (f == "Affirmative")) else s

df = pd.read_excel("geobias_report_64.xlsx")
df["aligned"] = [align(s, t, f) for s, t, f in zip(df["Score"], df["Topic"], df["Framing"])]
df["Origin"] = df["Model"].map(lambda m: g.PROVENANCE.get(m, "?"))
SHORT = {m: m.split("/")[-1] for m in g.MODELS}
df["M"] = df["Model"].map(SHORT)
ORDER = [SHORT[m] for m in g.MODELS]                      # EU, CN, US display order
TOPICS = list(g.TOPICS)
PROV_OF = {SHORT[m]: g.PROVENANCE[m] for m in g.MODELS}

# ===================== #1 VARIANCE DECOMPOSITION =====================
d = df.rename(columns={"Language": "Lang"})
ols = smf.ols("aligned ~ C(Origin)*C(Lang) + C(Framing) + C(Topic) "
              "+ C(Origin):C(Framing) + C(Lang):C(Topic) + C(Origin):C(Topic)", data=d).fit()
av = sm.stats.anova_lm(ols, typ=2)
av["pct_variance"] = (100 * av["sum_sq"] / av["sum_sq"].sum()).round(2)
av.to_csv("variance_decomposition.csv")

def r2(formula):
    return smf.ols(formula, data=d).fit().rsquared
r2_orig = r2("aligned ~ C(Origin)")
r2_ol = r2("aligned ~ C(Origin)+C(Lang)")
r2_olx = r2("aligned ~ C(Origin)*C(Lang)")
inc = {"origin_only": round(r2_orig,4),
       "+language": round(r2_ol,4),
       "+origin:language": round(r2_olx,4),
       "interaction_adds": round(r2_olx - r2_ol,4)}

# ===================== #2 LANGUAGE GATING =====================
piv = (df.groupby(["M","Topic","Language"])["aligned"].mean().unstack("Language"))
piv["shift"] = piv["Mandarin Chinese"] - piv["English"]      # negative => more Pro-China in ZH
gate = piv["shift"].unstack("Topic").reindex(ORDER)[TOPICS]
gate.to_csv("language_gating.csv")
lci = gate.mean(axis=1).round(2)                              # language-conditioning index

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(gate.values, cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(TOPICS))); ax.set_xticklabels([g.TOPICS[t]["short"] for t in TOPICS], rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([f"{m}" for m in ORDER], fontsize=9)
for i,m in enumerate(ORDER):
    ax.get_yticklabels()[i].set_color(g.PROV_COLOR[PROV_OF[m]])
for i in range(len(ORDER)):
    for j in range(len(TOPICS)):
        v = gate.values[i,j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 1.1 else "#333")
ax.set_title("Language-gating: stance shift English→Chinese (Δ = μ_ZH − μ_EN)\n"
             "red = becomes more Pro-China in Chinese", fontsize=11, fontweight="bold")
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Δ aligned score (− Pro-China)")
fig.tight_layout(); fig.savefig("fig_language_gating.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ===================== #5 IDEOLOGICAL PCA MAP =====================
# features: aligned mean per (topic x language) -> 14-dim profile per model
feat = (df.groupby(["M","Topic","Language"])["aligned"].mean()
          .unstack(["Topic","Language"]).reindex(ORDER))
X = feat.values
Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
PC = U[:, :2] * S[:2]
ve = (S**2 / (S**2).sum())[:2]
coords = pd.DataFrame({"Model": ORDER, "Origin": [PROV_OF[m] for m in ORDER],
                       "PC1": PC[:,0].round(3), "PC2": PC[:,1].round(3)})
coords.to_csv("pca_coordinates.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 7))
ax.axhline(0, color="#ccc", lw=.8); ax.axvline(0, color="#ccc", lw=.8)
for _, r in coords.iterrows():
    c = g.PROV_COLOR[r["Origin"]]
    ax.scatter(r["PC1"], r["PC2"], s=140, color=c, edgecolor="k", lw=.6, zorder=3)
    ax.annotate(r["Model"], (r["PC1"], r["PC2"]), fontsize=8, fontweight="bold",
                xytext=(6,4), textcoords="offset points", color=c)
ax.set_xlabel(f"PC1 ({100*ve[0]:.0f}% var)  —  ideological axis", fontsize=10)
ax.set_ylabel(f"PC2 ({100*ve[1]:.0f}% var)", fontsize=10)
ax.set_title("Ideological map of LLMs (PCA of topic×language stance profiles)", fontsize=12, fontweight="bold")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=g.PROV_COLOR[o],
          markersize=10,label=o) for o in ["EU","CN","US"]], loc="best", frameon=True)
fig.tight_layout(); fig.savefig("fig_ideological_map.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ===================== #6 CONVERGENCE TOPICS =====================
rows=[]
for t in TOPICS:
    s = df[df["Topic"]==t]
    mm = s.groupby("Model")["aligned"].mean()
    us = s[s["Origin"]=="US"]["aligned"].mean(); cn = s[s["Origin"]=="CN"]["aligned"].mean()
    rows.append(dict(Topic=t, pole=g.AFFIRMATIVE_POLE[t], between_model_sd=round(mm.std(),3),
                     US_group=round(us,2), CN_group=round(cn,2), US_minus_CN=round(us-cn,2)))
conv = pd.DataFrame(rows).sort_values("US_minus_CN")
conv.to_csv("topic_origin_effect.csv", index=False)

# ===================== #3 ACQUIESCENCE =====================
# yea-saying: mean RAW stance toward the proposition, regardless of pole.
df["raw_agree"] = df["Score"]   # +ve = agreed with whatever was asserted
acq = (df.groupby("M")["raw_agree"].mean().reindex(ORDER).round(3)
         .rename("acquiescence_mean_raw_agreement").reset_index())
acq["framing_swing"] = [round(abs(df[(df.M==m)&(df.Framing=="Affirmative")]["aligned"].mean()
                          - df[(df.M==m)&(df.Framing=="Reverse")]["aligned"].mean()),3) for m in ORDER]
acq.to_csv("acquiescence.csv", index=False)

# ===================== SUMMARY =====================
with open("ANALYSIS_SUMMARY.md","w",encoding="utf-8") as f:
    f.write("# Nature-Comms analyses — headline numbers\n\n")
    f.write("## #1 Variance decomposition (ANOVA, % of total SS)\n")
    f.write(av[["sum_sq","pct_variance"]].round(3).to_markdown()+"\n\n")
    f.write("Incremental R^2: " + str(inc) + "\n\n")
    f.write("## #2 Language-conditioning index (mean ZH-EN shift; - = more Pro-China in Chinese)\n")
    f.write(lci.to_markdown()+"\n\n")
    f.write("## #5 PCA ideological map\n")
    f.write(f"PC1 explains {100*ve[0]:.0f}%, PC2 {100*ve[1]:.0f}%\n\n")
    f.write(coords.to_markdown(index=False)+"\n\n")
    f.write("## #6 Topic origin effect (sorted by US-CN gap)\n")
    f.write(conv.to_markdown(index=False)+"\n\n")
    f.write("## #3 Acquiescence / framing swing\n")
    f.write(acq.to_markdown(index=False)+"\n")

print("VARIANCE DECOMPOSITION (% of total SS):")
print(av[["pct_variance"]].to_string())
print("\nIncremental R^2:", inc)
print("\nLanguage-conditioning index (ZH-EN, sorted):")
print(lci.sort_values().to_string())
print(f"\nPCA: PC1={100*ve[0]:.0f}%  PC2={100*ve[1]:.0f}%")
print("\nConvergence (topic origin effect):")
print(conv.to_string(index=False))
print("\nAcquiescence:")
print(acq.to_string(index=False))
print("\nsaved: variance_decomposition.csv, language_gating.csv, fig_language_gating.png,")
print("       pca_coordinates.csv, fig_ideological_map.png, topic_origin_effect.csv,")
print("       acquiescence.csv, ANALYSIS_SUMMARY.md")
