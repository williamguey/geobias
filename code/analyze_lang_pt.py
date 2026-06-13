# -*- coding: utf-8 -*-
"""Is the Mandarin pull corpus-specific or a generic non-English effect? Compare each
model's English->Portuguese shift (neutral third language) to its English->Mandarin shift."""
import pandas as pd, numpy as np
from scipy import stats
POLE={"Taiwan Conflict":"US","Trade and Tariffs":"US","South China Sea Dispute":"CN","Xinjiang Policies":"US",
      "Belt and Road Initiative":"CN","Technology and Semiconductors":"US","Dollar Dominance and BRICS":"CN"}
def align(s,t,f): return -s if ((POLE[t]=="CN")==(f=="Affirmative")) else s

main=pd.read_excel("geobias_report_64.xlsx")
main["M"]=main.Model.map(lambda m:m.split("/")[-1])
main["aligned"]=[align(s,t,f) for s,t,f in zip(main.Score,main.Topic,main.Framing)]
en=main[main.Language=="English"].groupby("M")["aligned"].mean()
zh=main[main.Language=="Mandarin Chinese"].groupby("M")["aligned"].mean()

pt_df=pd.read_excel("pilot_lang_pt.xlsx")
pt_df["M"]=pt_df.Model.map(lambda m:m.split("/")[-1])
pt=pt_df.groupby("M")["aligned"].mean()

C=pd.DataFrame({"EN":en,"ZH":zh,"PT":pt}).dropna()
C["shift_ZH"]=C.ZH-C.EN
C["shift_PT"]=C.PT-C.EN
print(C.round(3).to_string())
print()
print(f"mean EN->ZH shift: {C.shift_ZH.mean():+.3f}   (11/11 negative: {int((C.shift_ZH<0).sum())}/11)")
print(f"mean EN->PT shift: {C.shift_PT.mean():+.3f}   (negative: {int((C.shift_PT<0).sum())}/11)")
print(f"ratio |PT shift| / |ZH shift| (means): {abs(C.shift_PT.mean())/abs(C.shift_ZH.mean()):.2f}")
# is PT shift smaller in magnitude than ZH shift? paired across models
tt=stats.ttest_rel(C.shift_ZH.abs(), C.shift_PT.abs())
print(f"\npaired t, |ZH shift| vs |PT shift|: t={tt.statistic:.2f}, p={tt.pvalue:.2e}")
# is PT shift different from zero?
t0=stats.ttest_1samp(C.shift_PT,0); w0=stats.wilcoxon(C.shift_PT)
print(f"PT shift vs 0: t={t0.statistic:.2f}, p={t0.pvalue:.3f}; Wilcoxon p={w0.pvalue:.3f}")
print(f"ZH shift vs 0: t={stats.ttest_1samp(C.shift_ZH,0).statistic:.2f}, p={stats.ttest_1samp(C.shift_ZH,0).pvalue:.2e}")
C.round(3).to_csv("lang_pt_comparison.csv")

# figure: EN baseline, with PT and ZH shifts per model
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROV={"mistral-small-2603":"EU","deepseek-v4-flash":"CN","seed-2.0-lite":"CN","qwen3.6-plus":"CN",
      "minimax-m2.7":"CN","glm-5.1":"CN","gpt-5.3-chat":"US","gpt-4o-mini":"US","claude-sonnet-4.6":"US",
      "gemini-3.1-flash-lite":"US","grok-4.3":"US"}
COL={"EU":"#E8A33D","CN":"#C0392B","US":"#2E6FB5"}
order=C.sort_values("EN").index
fig,ax=plt.subplots(figsize=(9,5.6))
for i,m in enumerate(order):
    ax.plot([0,1,2],[C.loc[m,"EN"],C.loc[m,"PT"],C.loc[m,"ZH"]],color=COL[PROV[m]],lw=1.4,alpha=.8,marker="o",ms=5)
    ax.text(-0.05,C.loc[m,"EN"],m,ha="right",va="center",fontsize=7.5,color=COL[PROV[m]])
ax.axhline(0,color="#999",ls="--",lw=.8)
ax.set_xticks([0,1,2]); ax.set_xticklabels(["English","Portuguese\n(neutral)","Mandarin"],fontsize=10)
ax.set_xlim(-0.8,2.2); ax.set_ylabel("aligned stance (- Pro-China, + Pro-U.S.)",fontsize=10)
ax.set_title("The pull is corpus-specific: a neutral third language barely shifts stance,\nwhile Mandarin pulls every model toward Pro-China",fontsize=11,fontweight="bold")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("fig_S_language_pt.png",dpi=190,bbox_inches="tight")
print("\nsaved lang_pt_comparison.csv, fig_S_language_pt.png")
