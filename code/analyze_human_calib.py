# -*- coding: utf-8 -*-
"""Human-vs-judge calibration, done correctly. Pairwise on all valid rows (the EU judge
emitted 'Server refusal error' on some items; those drop only from the EU pair). NOTE the
sheet was deliberately enriched for judge-disagreement items, so agreement here is a
hard-case lower bound; the decisive metric is the direction-flip rate against the primary
(U.S.) judge, the judge that scored the paper's free-text responses."""
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

H=pd.read_excel("calibration_human_labels.xlsx")
K=pd.read_excel("calibration_key.xlsx")
LMAP={"strongly agree":2,"agree":1,"neutral":0,"disagree":-1,"strongly disagree":-2}
def m(s):
    return LMAP.get(str(s).strip().lower(), np.nan)
H["human"]=H["human_label"].map(m)
df=H[["id","Topic","Language","Framing","human"]].merge(K,on="id")
for who in ["US","CN","EU"]:
    df[who]=df[f"{who}_Cat"].map(m)
print("rows:", len(df), "| human valid:", df.human.notna().sum(),
      "| US valid:", df.US.notna().sum(), "| CN valid:", df.CN.notna().sum(),
      "| EU valid:", df.EU.notna().sum(), "(EU drops = server refusals)")

# how enriched for disagreement is this sheet?
v=df.dropna(subset=["US","CN"])
print(f"US-CN judge disagreements in this sheet: {int((v.US!=v.CN).sum())} of {len(v)} "
      f"(deliberately oversampled; population rate is ~2%)")

def flips(a,b):
    return int((((a>0)&(b<0))|((a<0)&(b>0))).sum())
def report(col,name):
    d=df.dropna(subset=["human",col])
    a,b=d["human"],d[col]
    ex=100*(a==b).mean(); k=cohen_kappa_score(a,b,weights="quadratic",labels=[-2,-1,0,1,2])
    print(f"  Human vs {name:3s} (n={len(d):3d}): exact {ex:4.1f}%  weighted kappa {k:.3f}  "
          f"direction flips {flips(a,b)} ({100*flips(a,b)/len(d):.1f}%)")

print("\n=== human vs each judge (raw agreement, all valid rows) ===")
report("US","US"); report("CN","CN"); report("EU","EU")

print("\n=== the metric that matters: does the human ever pick the OPPOSITE side from the primary (US) judge? ===")
d=df.dropna(subset=["human","US"])
fl=flips(d.human,d.US)
print(f"  On {len(d)} hard (disagreement-enriched) items, human and primary judge disagree on DIRECTION in {fl} cases ({100*fl/len(d):.1f}%).")
print(f"  The remaining differences are neutral-vs-stance or intensity, which do not change a stance's sign.")
# breakdown
def kind(h,j):
    if h==j: return "exact agree"
    if h==0 or j==0: return "neutral vs stance"
    if (h>0)!=(j>0): return "DIRECTION FLIP"
    return "intensity only"
print("\n  human vs primary judge, difference composition:")
print(pd.Series([kind(h,j) for h,j in zip(d.human,d.US)]).value_counts().to_string())
df.to_csv("human_calibration_merged.csv",index=False)
print("\nsaved human_calibration_merged.csv")
