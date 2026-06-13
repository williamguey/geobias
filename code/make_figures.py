"""
make_figures.py
===============
Builds the four main-text figures for the Results section, each anchored to one
claim. No API calls: everything is derived from files already on disk.

INPUT FILES (all produced earlier by the pipeline):
  - topic_model_stats.csv      per-topic, per-model aligned stats. Columns:
                               Topic, pole, Origin, Model, mu, sd, d,
                               mu_aff, mu_rev, mu_en, mu_zh, ref, neu
                               (mu is the polarity-aligned mean for that topic;
                                + = Pro-US, - = Pro-China)
  - topic_origin_effect.csv    per-topic US-minus-CN group gap (US_minus_CN),
                               used to order the heatmap columns by polarization.
  - variance_decomposition.csv ANOVA table; column pct_variance per factor.

DERIVED QUANTITIES (the logic):
  - "overall mu" per model      = mean of its 7 per-topic mu values. Because the
                                  design is balanced (equal cells per topic), the
                                  unweighted mean across topics equals the overall
                                  aligned mean across all 1,792 responses.
  - "overall English mu" / ZH   = mean of mu_en / mu_zh across the 7 topics.

THE FOUR FIGURES (claim each one supports):
  Fig 1  Origin ranking bar      -> Claim 1: origin predicts stance ASYMMETRICALLY
                                    (CN models form a tight Pro-China block; US
                                    models fan out across zero).
  Fig 2  EN->ZH slopegraph        -> Claim 2: querying in Mandarin pulls EVERY model
                                    toward Pro-China (every line slopes down).
  Fig 3  Model x Topic heatmap   -> Claim 3: origin alignment is topic-contingent
                                    (sovereignty columns split red/blue; the two
                                    economic columns converge).
  Fig 4  Variance partition bar  -> Claim 4: topic, origin and language are at
                                    near-parity among systematic factors; framing
                                    is negligible.

Run:  python make_figures.py     (from the geobias folder)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geobias_app as g

# ---- shared config pulled from the app so colours/order stay consistent ----
PROV_COLOR = g.PROV_COLOR                      # {"EU":gold, "CN":red, "US":blue}
ORDER = [m.split("/")[-1] for m in g.MODELS]   # display order: EU, then CN, then US
PROV_OF = {m.split("/")[-1]: g.PROVENANCE[m] for m in g.MODELS}
SHORT_TOPIC = {t: g.TOPICS[t]["short"] for t in g.TOPICS}

stats = pd.read_csv("topic_model_stats.csv")
stats["M"] = stats["Model"]                    # already short names in this file

# overall per-model aggregates (mean across the 7 topics)
overall = stats.groupby("M").agg(mu=("mu", "mean"),
                                 en=("mu_en", "mean"),
                                 zh=("mu_zh", "mean")).reindex(ORDER)

# Precise per-model overall stance, recomputed UNROUNDED from the master responses.
# (stats["mu"] is rounded to 2 d.p. per topic; averaging rounded values drifts the
# total, e.g. Gemini to -0.2857 instead of the true -0.2846. The bar annotations must
# match the per-model net values reported in the text, so use the unrounded series.)
def _align(s, t, f):
    return -s if ((g.AFFIRMATIVE_POLE[t] == "CN") == (f == "Affirmative")) else s
_raw = pd.read_excel("geobias_report_64.xlsx")
_raw["M"] = _raw["Model"].map(lambda m: m.split("/")[-1])
_raw["aligned"] = [_align(s, t, f) for s, t, f in zip(_raw["Score"], _raw["Topic"], _raw["Framing"])]
overall["mu_precise"] = _raw.groupby("M")["aligned"].mean().reindex(overall.index)


# =====================================================================
# FIGURE 1 — origin ranking bar (Claim 1: asymmetric origin alignment)
# =====================================================================
def fig1_origin():
    s = overall["mu_precise"].sort_values(ascending=True)   # most Pro-China at bottom
    colors = [PROV_COLOR[PROV_OF[m]] for m in s.index]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(range(len(s)), s.values, color=colors, edgecolor="black", lw=0.4)
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=9)
    for tick, m in zip(ax.get_yticklabels(), s.index):
        tick.set_color(PROV_COLOR[PROV_OF[m]])
    # value labels at the bar ends
    for i, v in enumerate(s.values):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8, color="#333")
    ax.set_xlim(-1.4, 0.9)
    ax.set_xlabel("Overall aligned preference score\n(← Pro-China      Pro-U.S. →)", fontsize=10)
    ax.set_title("Origin predicts stance asymmetrically:\nChinese models cohere, U.S. models disperse",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=PROV_COLOR[o],
              markersize=10, label=o) for o in ["EU", "CN", "US"]], loc="lower right", frameon=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig("fig1_origin_ranking.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# =====================================================================
# FIGURE 2 — English -> Mandarin slopegraph (Claim 2: universal language pull)
# =====================================================================
def fig2_language():
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for m in ORDER:
        c = PROV_COLOR[PROV_OF[m]]
        en, zh = overall.loc[m, "en"], overall.loc[m, "zh"]
        ax.plot([0, 1], [en, zh], color=c, lw=1.6, alpha=0.85, zorder=2)
        ax.scatter([0, 1], [en, zh], color=c, s=26, zorder=3)
        ax.text(-0.04, en, m, ha="right", va="center", fontsize=7.5, color=c)
    ax.axhline(0, color="#999", ls="--", lw=0.8, zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["English", "Mandarin"], fontsize=11)
    ax.set_xlim(-0.7, 1.15)
    ax.set_ylabel("Aligned preference score\n(← Pro-China      Pro-U.S. →)", fontsize=10)
    ax.set_title("Querying in Mandarin pulls every model toward Pro-China",
                 fontsize=11, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig("fig2_language_slopegraph.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# =====================================================================
# FIGURE 3 — model x topic heatmap (Claim 3: topic-contingent alignment)
# =====================================================================
def fig3_heatmap():
    # columns ordered from most polarizing to most convergent (US-CN gap desc)
    eff = pd.read_csv("topic_origin_effect.csv").sort_values("US_minus_CN", ascending=False)
    topic_order = list(eff["Topic"])
    mat = stats.pivot(index="M", columns="Topic", values="mu").reindex(ORDER)[topic_order]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    im = ax.imshow(mat.values, cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(topic_order)))
    ax.set_xticklabels([SHORT_TOPIC[t] for t in topic_order], rotation=32, ha="right", fontsize=8)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(ORDER, fontsize=9)
    for tick, m in zip(ax.get_yticklabels(), ORDER):
        tick.set_color(PROV_COLOR[PROV_OF[m]])
    for i in range(len(ORDER)):
        for j in range(len(topic_order)):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 1.1 else "#222")
    # group separator lines between EU/CN/US blocks
    provs = [PROV_OF[m] for m in ORDER]
    for i in range(1, len(provs)):
        if provs[i] != provs[i - 1]:
            ax.axhline(i - 0.5, color="black", lw=1.2)
    ax.set_title("Origin alignment is topic-contingent:\nsovereignty topics polarize, economic topics converge",
                 fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Aligned preference score (← Pro-China | Pro-U.S. →)", fontsize=8)
    fig.tight_layout(); fig.savefig("fig3_topic_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# =====================================================================
# FIGURE 4 — variance partition (Claim 4: topic ~ origin ~ language >> framing)
# =====================================================================
def fig4_variance():
    av = pd.read_csv("variance_decomposition.csv", index_col=0)
    # readable factor names; drop Residual from the bars (shown as a caption note)
    rename = {"C(Origin)": "Origin", "C(Lang)": "Language", "C(Framing)": "Framing",
              "C(Topic)": "Topic", "C(Origin):C(Lang)": "Origin x Language",
              "C(Origin):C(Framing)": "Origin x Framing", "C(Lang):C(Topic)": "Language x Topic",
              "C(Origin):C(Topic)": "Origin x Topic"}
    resid = float(av.loc["Residual", "pct_variance"])
    av = av.drop(index="Residual").rename(index=rename)
    s = av["pct_variance"].sort_values(ascending=True)
    # highlight the three near-parity main effects
    main3 = {"Topic", "Origin", "Language"}
    colors = ["#2E6FB5" if name in main3 else "#b9b9b9" for name in s.index]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(range(len(s)), s.values, color=colors, edgecolor="black", lw=0.4)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=9)
    for i, v in enumerate(s.values):
        ax.text(v + 0.08, i, f"{v:.2f}%", va="center", fontsize=8, color="#333")
    ax.set_xlabel("Share of total variance explained (%)", fontsize=10)
    ax.set_title("Topic, origin and language sit at near-parity;\nframing is negligible as a population effect",
                 fontsize=11, fontweight="bold")
    ax.text(0.98, 0.04, f"residual = {resid:.2f}% (between-model + interactions + iteration noise), not shown",
            transform=ax.transAxes, ha="right", fontsize=7.5, style="italic", color="#666")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig("fig4_variance_partition.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig1_origin();    print("wrote fig1_origin_ranking.png")
    fig2_language();  print("wrote fig2_language_slopegraph.png")
    fig3_heatmap();   print("wrote fig3_topic_heatmap.png")
    fig4_variance();  print("wrote fig4_variance_partition.png")
