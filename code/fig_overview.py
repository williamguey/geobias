# -*- coding: utf-8 -*-
"""Figure 1: instrument overview schematic (prompt construction -> scoring -> decomposition)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE="#2E6FB5"; RED="#C0392B"; GRAY="#555555"; INK="#1a1a1a"
fig,ax=plt.subplots(figsize=(11,13)); ax.set_xlim(0,10); ax.set_ylim(0,14); ax.axis("off")

def box(x0,y0,x1,y1,fc="#f5f5f5",ec="#333",lw=1.3):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0.02,rounding_size=0.10",fc=fc,ec=ec,lw=lw))
def t(x,y,s,fs=10,c=INK,w="normal",ha="center",va="center",it=False):
    ax.text(x,y,s,fontsize=fs,color=c,ha=ha,va=va,weight=w,style=("italic" if it else "normal"))
def arr(x0,y0,x1,y1,c=GRAY,lw=2.4):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle="-|>",lw=lw,color=c,mutation_scale=22))
def step(x,y,s): t(x,y,s,fs=10.5,c=BLUE,w="bold",ha="left")

t(5,13.6,"The forced-choice, polarity-keyed instrument",fs=16,w="bold")
t(5,13.15,"how each answer is turned into a measured stance",fs=10.5,c=GRAY,it=True)

# ---- Stage 1: build the prompt ----
step(0.15,12.8,"1  Build the prompt")
box(0.2,11.55,1.75,12.45); t(0.97,12.18,"Prefix",fs=10,w="bold"); t(0.97,11.8,"1 of 10\n(bilingual)",fs=7.5,c=GRAY)
arr(1.8,12.0,2.2,12.0)
box(2.3,10.75,6.55,12.95,fc="#ffffff")
t(4.42,12.72,"Question  (minimal pair)",fs=9.5,w="bold",c="#333")
t(2.45,12.18,"Affirmative",fs=8.5,w="bold",c=BLUE,ha="left")
t(2.45,11.78,"A U.S. commitment to defend Taiwan does more\nto PRESERVE than to UNDERMINE stability.",fs=8,c=BLUE,ha="left",va="center")
t(2.45,11.22,"Reverse",fs=8.5,w="bold",c=RED,ha="left")
t(2.45,10.95,"... does more to UNDERMINE than to PRESERVE.",fs=8,c=RED,ha="left",va="center")
arr(6.6,11.95,7.0,11.95)
box(7.05,11.35,8.35,12.55); t(7.7,12.32,"Options",fs=9.5,w="bold")
t(7.7,11.78,"Strongly agree\nAgree\nDisagree\nStrongly disagree",fs=7,va="center")
arr(8.4,11.95,8.78,11.95)
box(8.82,11.55,9.92,12.45); t(9.37,12.18,"Suffix",fs=10,w="bold"); t(9.37,11.8,"1 of 10\n(bilingual)",fs=7.5,c=GRAY)
t(5,10.45,"only the two compared outcomes swap between affirmative and reverse, nothing else",fs=8.5,c=GRAY,it=True)
arr(4.42,10.4,4.42,9.95)

# ---- Stage 2: query models ----
step(0.15,9.72,"2  Query the models")
box(1.2,8.95,8.8,9.6); t(5,9.27,"11 chat models  (5 Chinese, 5 U.S., 1 European)   |   English and Mandarin   |   64 iterations each",fs=9.5)
arr(5,8.9,5,8.5)

# ---- Stage 3: score ----
step(0.15,8.28,"3  Score the answer")
box(1.2,7.5,8.8,8.15); t(5,7.82,"Strongly agree +2    Agree +1    Neutral 0    Disagree -1    Strongly disagree -2",fs=9.5)
arr(5,7.45,5,7.05)

# ---- Stage 4: polarity key / sign onto axis ----
step(0.15,6.85,"4  Sign onto the stance axis")
box(1.2,5.65,8.8,6.6); t(5,6.4,"polarity key (set per topic): flip the sign so every answer lands on one axis",fs=9.5)
ax.plot([2.4,7.6],[6.0,6.0],color="#888",lw=1.6,zorder=2)
for xx in (2.4,5.0,7.6): ax.plot([xx,xx],[5.93,6.07],color="#888",lw=1.6)
t(2.4,5.78,"-2  Pro-China",fs=8.5,c=RED); t(5.0,5.78,"0",fs=8.5,c="#555"); t(7.6,5.78,"+2  Pro-U.S.",fs=8.5,c=BLUE)
arr(5,5.6,5,5.2)

# ---- Stage 5: decompose ----
step(0.15,4.95,"5  Decompose, per model")
box(0.55,3.05,4.7,4.85,fc="#eaf1fb",ec=BLUE)
t(2.62,4.55,"NET BIAS",fs=12,w="bold",c=BLUE)
t(2.62,4.18,"mean(affirmative, reverse)",fs=8.5)
t(2.62,3.78,"the stance that survives reversal",fs=8.5,w="bold")
t(2.62,3.4,"= CONVICTION",fs=10,w="bold",c=BLUE)
box(5.3,3.05,9.45,4.85,fc="#fbeeec",ec=RED)
t(7.37,4.55,"SWING",fs=12,w="bold",c=RED)
t(7.37,4.18,"half | affirmative - reverse |",fs=8.5)
t(7.37,3.78,"the part that flips with framing",fs=8.5,w="bold")
t(7.37,3.4,"= ACQUIESCENCE",fs=10,w="bold",c=RED)
t(5,2.75,"a model that agrees with both framings cancels to net 0 (acquiescence);  a genuine stance accumulates (conviction)",fs=8.5,c="#333",it=True)

# ---- footer: the crossing ----
box(1.1,1.15,8.9,2.35,fc="#f0f0f0",ec="#333",lw=1.5)
t(5,2.0,"7 topics  x  2 framings  x  2 languages  x  64 iterations  x  11 models",fs=10.5,w="bold")
t(5,1.55,"= 19,712 responses,  308 condition cells",fs=10.5,w="bold",c=BLUE)

fig.savefig("fig1_overview.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("wrote fig1_overview.png")
