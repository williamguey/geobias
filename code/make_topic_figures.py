# -*- coding: utf-8 -*-
"""Regenerate the per-topic hierarchical stance figures (Supplementary Figs. S5-S11; the
panels that also compose main-text Fig. 5) from the saved responses. No API calls.

Run from the data directory so the input/outputs resolve there:
    cd data && python ../code/make_topic_figures.py
"""
import pandas as pd
import geobias_app as g

df = pd.read_excel("geobias_report_64.xlsx")
for topic in g.TOPICS:
    sub = df[df["Topic"] == topic]
    if len(sub):
        out = g.make_topic_figure(topic, sub, g.MODELS, f"fig_{topic.replace(' ', '_')}.png")
        print("wrote", out)
