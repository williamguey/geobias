# -*- coding: utf-8 -*-
"""Analyze the transfer pilot: per-domain net/swing, and whether acquiescence (swing)
is a stable model trait across the geopolitical and the new domains."""
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_excel("pilot_transfer.xlsx")
df["M"] = df.Model.map(lambda m: m.split("/")[-1])

def net_swing(s):
    aff = s[s.Framing == "Affirmative"]["aligned"].mean()
    rev = s[s.Framing == "Reverse"]["aligned"].mean()
    return (aff + rev) / 2, abs(aff - rev) / 2

print("=== per-DOMAIN net bias and swing ===")
for dom in df.Domain.unique():
    print(f"\n-- {dom} --")
    rows = []
    for m in df.M.unique():
        s = df[(df.M == m) & (df.Domain == dom)]
        net, sw = net_swing(s)
        rows.append((m, round(net, 3), round(sw, 3), round(100*s.IsNeutral.mean(),1)))
    r = pd.DataFrame(rows, columns=["model","net","swing","neu%"]).sort_values("swing", ascending=False)
    print(r.to_string(index=False))

# cross-domain stability of swing: geopolitics vs new domains
geo = pd.read_csv("bias_decomposition.csv")          # has swing_half (geopolitics)
geo_sw = dict(zip(geo.model, geo.swing_half))
new = []
for m in df.M.unique():
    _, sw = net_swing(df[df.M == m])
    new.append((m, geo_sw.get(m, np.nan), sw))
C = pd.DataFrame(new, columns=["model","geo_swing","new_swing"]).dropna()
pear = stats.pearsonr(C.geo_swing, C.new_swing)
spear = stats.spearmanr(C.geo_swing, C.new_swing)
print("\n=== acquiescence stability across domains (per-model swing) ===")
print(C.sort_values("new_swing", ascending=False).to_string(index=False))
print(f"\nPearson r(geo_swing, new_swing) = {pear[0]:+.3f} (p={pear[1]:.3f})")
print(f"Spearman rho = {spear[0]:+.3f} (p={spear[1]:.3f})")
print(f"Heaviest acquiescer in geopolitics: {C.sort_values('geo_swing').iloc[-1]['model']}")
print(f"Heaviest acquiescer in new domains: {C.sort_values('new_swing').iloc[-1]['model']}")

# raw agreement vs net bias (does raw agreement misrank?)
print("\n=== raw agreement vs net bias (new domains, pooled) ===")
rows = []
for m in df.M.unique():
    s = df[df.M == m]
    net, sw = net_swing(s)
    rows.append((m, round(s.Score.mean(),3), round(net,3)))
rr = pd.DataFrame(rows, columns=["model","raw_agree","net_bias"])
print(f"corr(raw_agree, net_bias) = {np.corrcoef(rr.raw_agree, rr.net_bias)[0,1]:+.3f} (should be weak if raw is a poor proxy)")
