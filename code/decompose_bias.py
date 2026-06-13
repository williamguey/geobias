"""Decompose each model into (a) net directional bias and (b) framing-driven swing.
Question: do the high-swing ('acquiescent') models STILL carry a net bias direction?
net  = (mean aligned under Affirmative + mean aligned under Reverse)/2  -> directional bias
swing/2 = (Affirmative - Reverse)/2                                     -> framing susceptibility
The two are orthogonal: a model can swing hard AND be biased."""
import numpy as np, pandas as pd
from scipy import stats

POLE = {"Taiwan Conflict":"US","Trade and Tariffs":"US","South China Sea Dispute":"CN",
        "Xinjiang Policies":"US","Belt and Road Initiative":"CN",
        "Technology and Semiconductors":"US","Dollar Dominance and BRICS":"CN"}
PROV = {"mistralai/mistral-small-2603":"EU","deepseek/deepseek-v4-flash":"CN",
        "bytedance-seed/seed-2.0-lite":"CN","qwen/qwen3.6-plus":"CN",
        "minimax/minimax-m2.7":"CN","z-ai/glm-5.1":"CN","openai/gpt-5.3-chat":"US",
        "openai/gpt-4o-mini":"US","anthropic/claude-sonnet-4.6":"US",
        "google/gemini-3.1-flash-lite":"US","x-ai/grok-4.3":"US"}
def align(s,t,f): return -s if ((POLE[t]=="CN")==(f=="Affirmative")) else s

df = pd.read_excel("geobias_report_64.xlsx")
df["M"] = df.Model.map(lambda m: m.split("/")[-1])
df["aligned"] = [align(s,t,f) for s,t,f in zip(df.Score, df.Topic, df.Framing)]

rows=[]
for m in df.M.unique():
    sub = df[df.M==m]
    aff = sub[sub.Framing=="Affirmative"]["aligned"]
    rev = sub[sub.Framing=="Reverse"]["aligned"]
    net = (aff.mean()+rev.mean())/2            # directional bias (acquiescence-robust)
    acq = abs(aff.mean()-rev.mean())/2          # framing susceptibility (half-swing)
    raw_agree = sub.Score.mean()                # +ve = yea-sayer, -ve = nay-sayer
    # is the net bias significantly different from 0? CELL-LEVEL test: the cell (topic x
    # language x framing, 28 per model) is the unit of inference, never the responses.
    cells = sub.groupby(["Topic","Language","Framing"])["aligned"].mean()   # 28 cell means
    t,p = stats.ttest_1samp(cells, 0.0)
    try:
        wp = stats.wilcoxon(cells).pvalue
    except Exception:
        wp = float("nan")
    # ROBUSTNESS: 14 framing-collapsed net values (topic x language) -- the natural unit for
    # a quantity defined as the average across framings. Reported as a robustness check.
    net14 = cells.groupby(level=[0,1]).mean()              # 14 net values
    t14,p14 = stats.ttest_1samp(net14, 0.0)
    try:
        wp14 = stats.wilcoxon(net14).pvalue
    except Exception:
        wp14 = float("nan")
    rows.append((m, PROV[sub.Model.iloc[0]], round(net,3), round(acq,3),
                 round(raw_agree,3), "Pro-CN" if net<0 else "Pro-US",
                 "yes" if p<0.05 else "no", f"{p:.2e}", f"{wp:.2e}",
                 f"{p14:.2e}", f"{wp14:.2e}", p))
res = pd.DataFrame(rows, columns=["model","origin","NET_bias","swing_half","raw_agree",
                                  "direction","biased_p<.05","p_t_cell","p_wilcoxon_cell",
                                  "p_t_net14","p_wilcoxon_net14","_p28"])
# Holm correction across the 11 per-model 28-cell t-tests
order = res["_p28"].sort_values().index
nT = len(res); holm = {}; running = 0.0
for rank,idx in enumerate(order):
    adj = min((nT-rank)*res.loc[idx,"_p28"], 1.0)
    running = max(running, adj); holm[idx] = running
res["holm_p28"] = res.index.map(holm).map(lambda x: f"{x:.2e}")
res["holm_sig.05"] = res.index.map(holm).map(lambda x: "yes" if x<0.05 else "no")
res = res.drop(columns=["_p28"])
res = res.reindex(res.NET_bias.abs().sort_values(ascending=False).index)
pd.set_option("display.width",160)
print(res.to_string(index=False))

# focus: the four models that FAILED the reversal test
print("\n--- the four high-swing models: do they still carry a net direction? ---")
swingers = ["glm-5.1","gemini-3.1-flash-lite","qwen3.6-plus","mistral-small-2603"]
print(res[res.model.isin(swingers)].to_string(index=False))
res.to_csv("bias_decomposition.csv", index=False)
print("\nsaved bias_decomposition.csv")
