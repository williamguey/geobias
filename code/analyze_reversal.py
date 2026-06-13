"""Does dual framing prove the forced choice is a genuine stance, not an arbitrary pick?
Test: agreeing with the affirmative proposition (+) must mean DISAGREEING with the
reversed proposition (-). If the choice were random/arbitrary, reversing the wording
would NOT reverse the raw answer. All from real data; no API."""
import numpy as np, pandas as pd

POLE = {"Taiwan Conflict":"US","Trade and Tariffs":"US","South China Sea Dispute":"CN",
        "Xinjiang Policies":"US","Belt and Road Initiative":"CN",
        "Technology and Semiconductors":"US","Dollar Dominance and BRICS":"CN"}
PROV = {"mistralai/mistral-small-2603":"EU","deepseek/deepseek-v4-flash":"CN",
        "bytedance-seed/seed-2.0-lite":"CN","qwen/qwen3.6-plus":"CN",
        "minimax/minimax-m2.7":"CN","z-ai/glm-5.1":"CN","openai/gpt-5.3-chat":"US",
        "openai/gpt-4o-mini":"US","anthropic/claude-sonnet-4.6":"US",
        "google/gemini-3.1-flash-lite":"US","x-ai/grok-4.3":"US"}

def align(s, t, f):
    return -s if ((POLE[t]=="CN")==(f=="Affirmative")) else s

df = pd.read_excel("geobias_report_64.xlsx")
df["M"] = df["Model"].map(lambda m: m.split("/")[-1])
df["aligned"] = [align(s,t,f) for s,t,f in zip(df.Score, df.Topic, df.Framing)]

# ---------- TEST 1: raw reversal (the user's exact claim) ----------
# per Model x Topic x Language: mean raw agreement under each framing
g = (df.groupby(["M","Topic","Language","Framing"])["Score"].mean()
       .unstack("Framing"))
g = g.dropna()
a, r = g["Affirmative"].values, g["Reverse"].values
# consider only cells where the model actually takes a side under at least one framing
mask = (np.abs(a) > 0.25) | (np.abs(r) > 0.25)
opp = np.sign(a[mask]) != np.sign(r[mask])
print("=== TEST 1: RAW REVERSAL (does asking the reverse flip the raw answer?) ===")
print(f"cells with a real side taken: {mask.sum()} of {len(a)}")
print(f"  reversal WORKS (opposite raw sign aff vs reverse): {opp.mean()*100:.1f}% of those cells")
print(f"  corr(raw affirmative, raw reverse) across all {len(a)} cells: {np.corrcoef(a,r)[0,1]:+.3f}  (want strongly NEGATIVE)")

# ---------- TEST 2: aligned consistency ----------
# after polarity alignment, a genuine stance gives the SAME score under both framings.
ga = (df.groupby(["M","Topic","Framing"])["aligned"].mean().unstack("Framing")).dropna()
Aa, Ar = ga["Affirmative"].values, ga["Reverse"].values
print("\n=== TEST 2: ALIGNED CONSISTENCY (aff vs reverse agree after keying) ===")
print(f"  corr(aligned aff, aligned reverse) over {len(Aa)} model x topic cells: {np.corrcoef(Aa,Ar)[0,1]:+.3f}  (want ~ +1)")
slope = np.polyfit(Aa, Ar, 1)[0]
print(f"  regression slope aligned_reverse ~ aligned_aff: {slope:+.3f}  (want ~ +1)")

# ---------- TEST 3: non-randomness via reproducibility ----------
# within each Model x Topic x Language x Framing cell (64 iters), how consistent?
cell = df.groupby(["M","Topic","Language","Framing"])["Score"].agg(["mean","std"]).dropna()
cell["absd"] = cell["mean"].abs() / cell["std"].replace(0, np.nan)
# random forced-choice null over {-2,-1,1,2} uniform: SD ~ 1.58; over full {-2..2}: ~1.41
print("\n=== TEST 3: REPRODUCIBILITY (random pick would give SD ~1.4-1.6, |d|~0) ===")
print(f"  median within-cell SD across 64 iters: {cell['mean'].pipe(lambda x: cell['std'].median()):.2f}  (random null ~1.4-1.6)")
print(f"  cells with |Cohen d| > 1 (clearly non-random stance): {(cell['absd']>1).mean()*100:.1f}%")
print(f"  cells with |Cohen d| > 2 (near-deterministic stance):  {(cell['absd']>2).mean()*100:.1f}%")

# ---------- PER-MODEL verdict ----------
print("\n=== PER-MODEL: does the reversal test pass? ===")
rows=[]
for m in df.M.unique():
    sub = df[df.M==m]
    gg = sub.groupby(["Topic","Language","Framing"])["Score"].mean().unstack("Framing").dropna()
    aa, rr = gg["Affirmative"].values, gg["Reverse"].values
    msk = (np.abs(aa)>0.25)|(np.abs(rr)>0.25)
    rev = (np.sign(aa[msk])!=np.sign(rr[msk])).mean()*100 if msk.sum() else np.nan
    gga = sub.groupby(["Topic","Framing"])["aligned"].mean().unstack("Framing").dropna()
    cons = np.corrcoef(gga["Affirmative"], gga["Reverse"])[0,1]
    swing = abs(sub[sub.Framing=="Affirmative"].aligned.mean()-sub[sub.Framing=="Reverse"].aligned.mean())
    neu = sub.IsNeutral.mean()*100
    rows.append((m, PROV[sub.Model.iloc[0]], rev, cons, swing, neu))
res = pd.DataFrame(rows, columns=["model","origin","raw_reversal_%","aligned_consistency_r","framing_swing","neutral_%"])
res = res.sort_values("framing_swing")
pd.set_option("display.width",160)
print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
res.to_csv("reversal_test.csv", index=False)
print("\nsaved reversal_test.csv")
