"""Persist every number cited in the Methods 'judge validation' text into
durable files: pool_split.csv, judge_validation_metrics.csv,
judge_disagreement_breakdown.csv, and a human-readable VALIDATION_SUMMARY.md.
All recomputed from the saved raw-label files (no API)."""
import re
import numpy as np
import pandas as pd
import geobias_app as g

# ---------- helpers ----------
def align(score, topic, framing):
    aff_is_cn = g.AFFIRMATIVE_POLE[topic] == "CN"
    return -score if (aff_is_cn == (framing == "Affirmative")) else score

def qwk(a, b, cats=(-2,-1,0,1,2)):
    k=len(cats); idx={c:i for i,c in enumerate(cats)}
    O=np.zeros((k,k))
    for x,y in zip(a,b): O[idx[x],idx[y]]+=1
    W=np.array([[((i-j)**2)/((k-1)**2) for j in range(k)] for i in range(k)])
    E=np.outer(O.sum(1),O.sum(0))/O.sum(); den=(W*E).sum()
    return 1-(W*O).sum()/den if den>0 else float("nan")

def fleiss(rows, cats=(-2,-1,0,1,2)):
    k=len(cats); idx={c:i for i,c in enumerate(cats)}
    N=len(rows); n=len(rows[0]); M=np.zeros((N,k))
    for i,r in enumerate(rows):
        for s in r: M[i,idx[s]]+=1
    p_j=M.sum(0)/(N*n); P_i=(np.square(M).sum(1)-n)/(n*(n-1))
    return (P_i.mean()-np.square(p_j).sum())/(1-np.square(p_j).sum())

# ---------- 1. pool split (from full dataset) ----------
full = pd.read_excel("geobias_report_64.xlsx")
LABELS={"strongly agree","agree","disagree","strongly disagree","neutral",
        "强烈同意","同意","不同意","强烈不同意","中立"}
norm=lambda s: re.sub(r"[\s。．.,!！、]+$","",s.strip()).lower() if isinstance(s,str) else ""
def kind(r):
    if not isinstance(r,str) or r.strip()=="" : return "empty"
    if r.strip()==g.REFUSAL_TOKEN: return "pipeline_refusal"
    return "compliant" if norm(r) in LABELS else "free_text"
full["kind"]=full["Raw"].map(kind)
pool=full["kind"].value_counts().rename_axis("kind").reset_index(name="count")
pool["pct"]=(100*pool["count"]/len(full)).round(2)
pool.to_csv("pool_split.csv", index=False)

# ---------- 2. validation metrics (from 3-way file) ----------
ft=pd.read_excel("judge_validation_3way.xlsx")
for who in ("US","CN","EU"):
    ft[f"{who}_aligned"]=[align(s,t,f) for s,t,f in zip(ft[f"{who}_score"],ft["Topic"],ft["Framing"])]
ft["Origin"]=ft["Model"].map(lambda m: g.PROVENANCE.get(m,"?"))

def pair_metrics(x,y,ax,ay,sub=None):
    d = ft if sub is None else sub
    return dict(n=len(d),
               exact=round(100*(d[x]==d[y]).mean(),1),
               weighted_kappa=round(qwk(list(d[x]),list(d[y])),3),
               signed_drift=round((d[ay]-d[ax]).mean(),3))

rows=[]
for nm,x,y,ax,ay in [("US-CN","US_score","CN_score","US_aligned","CN_aligned"),
                     ("US-EU","US_score","EU_score","US_aligned","EU_aligned"),
                     ("CN-EU","CN_score","EU_score","CN_aligned","EU_aligned")]:
    m=pair_metrics(x,y,ax,ay); m["pair"]=nm; rows.append(m)
overall=pd.DataFrame(rows)[["pair","n","exact","weighted_kappa","signed_drift"]]
overall.to_csv("judge_validation_metrics.csv", index=False)

fleiss_all=round(fleiss(list(zip(ft["US_score"],ft["CN_score"],ft["EU_score"]))),3)
means={w:round(ft[f"{w}_aligned"].mean(),3) for w in ("US","CN","EU")}

# per-stratum US-CN (the decisive pair)
strata=[]
for t in g.TOPICS: strata.append(("topic",t,ft[ft["Topic"]==t]))
for l in ["English","Mandarin Chinese"]: strata.append(("language",l,ft[ft["Language"]==l]))
for fr in ["Affirmative","Reverse"]: strata.append(("framing",fr,ft[ft["Framing"]==fr]))
strata.append(("hot_cell","ZH x Reverse",
               ft[(ft["Language"]=="Mandarin Chinese")&(ft["Framing"]=="Reverse")]))
srows=[]
for typ,name,sub in strata:
    if len(sub)<2: continue
    srows.append(dict(stratum_type=typ, stratum=name, **pair_metrics("US_score","CN_score","US_aligned","CN_aligned",sub)))
pd.DataFrame(srows).to_csv("judge_validation_by_stratum.csv", index=False)

# disagreement-type breakdown
def dkind(a,b):
    if a==b: return "agree"
    if a==0 or b==0: return "neutral_vs_stance"
    if (a>0)!=(b>0): return "direction_flip"
    return "intensity_only"
drows=[]
for nm,x,y in [("US-CN","US_score","CN_score"),("US-EU","US_score","EU_score"),("CN-EU","CN_score","EU_score")]:
    vc=pd.Series([dkind(a,b) for a,b in zip(ft[x],ft[y])]).value_counts()
    drows.append(dict(pair=nm, **{k:int(vc.get(k,0)) for k in ["agree","neutral_vs_stance","intensity_only","direction_flip"]}))
pd.DataFrame(drows).to_csv("judge_disagreement_breakdown.csv", index=False)

# ---------- 3. human-readable summary ----------
pj={r["kind"]:(r["count"],r["pct"]) for _,r in pool.iterrows()}
with open("VALIDATION_SUMMARY.md","w",encoding="utf-8") as f:
    f.write("# Judge validation — summary of cited numbers\n\n")
    f.write("Recomputed from saved raw files; reproduce with `build_validation_artifacts.py`.\n\n")
    f.write("## Source files\n")
    f.write("- `geobias_report_64.xlsx` — full dataset (19,712 rows; primary judge gpt-4o-mini)\n")
    f.write("- `judge_validation_cn.xlsx` — free-text rows + US & CN judge labels\n")
    f.write("- `judge_validation_3way.xlsx` — same rows + EU judge labels\n\n")
    f.write(f"## Response-kind split (n={len(full)})\n")
    for k in ["compliant","free_text","pipeline_refusal"]:
        if k in pj: f.write(f"- {k}: {pj[k][0]} ({pj[k][1]}%)\n")
    f.write("\n## Pairwise judge agreement (free-text pool, n=%d)\n" % len(ft))
    f.write(overall.to_markdown(index=False))
    f.write(f"\n\nFleiss' kappa (all 3): {fleiss_all}\n")
    f.write(f"\nMean aligned stance: US={means['US']}, CN={means['CN']}, EU={means['EU']}\n")
    f.write("\n## Disagreement composition\n")
    f.write(pd.DataFrame(drows).to_markdown(index=False))
    f.write("\n\n## US-CN by stratum\n")
    f.write(pd.DataFrame(srows).to_markdown(index=False))
    f.write("\n")

print("wrote: pool_split.csv, judge_validation_metrics.csv, judge_validation_by_stratum.csv,")
print("       judge_disagreement_breakdown.csv, VALIDATION_SUMMARY.md")
print(f"\nFleiss={fleiss_all}  means={means}")
print(overall.to_string(index=False))
