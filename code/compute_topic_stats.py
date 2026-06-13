"""Per-topic, per-model statistics for writing the Results (no API).
All scores polarity-aligned: + = Pro-US, - = Pro-China."""
import numpy as np, pandas as pd
import geobias_app as g

def align(s, t, f):
    return -s if ((g.AFFIRMATIVE_POLE[t] == "CN") == (f == "Affirmative")) else s

df = pd.read_excel("geobias_report_64.xlsx")
df["aligned"] = [align(s, t, f) for s, t, f in zip(df["Score"], df["Topic"], df["Framing"])]
df["Origin"] = df["Model"].map(lambda m: g.PROVENANCE.get(m, "?"))
SHORT = {m: m.split("/")[-1] for m in g.MODELS}

def fmean(s, col, val):
    sel = s[s[col] == val]["aligned"]
    return round(sel.mean(), 2) if len(sel) else float("nan")

rows = []
for topic in g.TOPICS:
    sub_t = df[df["Topic"] == topic]
    for m in g.MODELS:
        s = sub_t[sub_t["Model"] == m]
        al = s["aligned"].values
        mu, sd = al.mean(), al.std(ddof=1)
        d = mu / sd if sd > 0 else np.nan
        rows.append(dict(
            Topic=topic, pole=g.AFFIRMATIVE_POLE[topic], Origin=g.PROVENANCE[m], Model=SHORT[m],
            mu=round(mu, 2), sd=round(sd, 2), d=(round(d, 2) if d == d else 99.0),
            mu_aff=fmean(s, "Framing", "Affirmative"), mu_rev=fmean(s, "Framing", "Reverse"),
            mu_en=fmean(s, "Language", "English"), mu_zh=fmean(s, "Language", "Mandarin Chinese"),
            ref=round(100 * s["IsRefusal"].mean(), 1), neu=round(100 * s["IsNeutral"].mean(), 1)))

stats = pd.DataFrame(rows)
stats.to_csv("topic_model_stats.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 20)
for topic in g.TOPICS:
    st = stats[stats["Topic"] == topic].copy()
    pole = st["pole"].iloc[0]
    grp = df[df["Topic"] == topic].groupby("Origin")["aligned"].mean().round(2).to_dict()
    framesens = df[df["Topic"] == topic].groupby(["Origin", "Framing"])["aligned"].mean().unstack()
    print("\n" + "=" * 100)
    print(f"TOPIC: {topic}   (affirmative pole = {pole})   group means: {grp}")
    print("=" * 100)
    print(st.sort_values("mu")[["Origin", "Model", "mu", "sd", "d", "mu_aff", "mu_rev", "mu_en", "mu_zh", "ref", "neu"]].to_string(index=False))

# overall ranking across topics
print("\n" + "#" * 100)
print("OVERALL (mean aligned across all 7 topics)")
ov = df.groupby("Model")["aligned"].mean().round(2).sort_values()
for m, v in ov.items():
    print(f"  {g.PROVENANCE[m]} {SHORT[m]:28s} {v:+.2f}")
print("\ngroup overall:", df.groupby("Origin")["aligned"].mean().round(3).to_dict())
# framing sensitivity per group (mean |aff-rev| across models/topics)
df2 = df.groupby(["Origin","Model","Topic","Framing"])["aligned"].mean().unstack()
df2["sens"] = (df2["Affirmative"] - df2["Reverse"]).abs()
print("framing sensitivity (mean |aff-rev|) by origin:",
      df2.reset_index().groupby("Origin")["sens"].mean().round(3).to_dict())
# language sensitivity
dl = df.groupby(["Origin","Model","Topic","Language"])["aligned"].mean().unstack()
dl["lsens"] = (dl["English"] - dl["Mandarin Chinese"]).abs()
print("language sensitivity (mean |EN-ZH|) by origin:",
      dl.reset_index().groupby("Origin")["lsens"].mean().round(3).to_dict())
