# Geopolitical bias in large language models: forced-choice, polarity-keyed instrument

Data and code for the study of geopolitical stance in large language models using a
forced-choice, polarity-keyed dual-framing instrument that separates **conviction**
(net bias, the stance that survives reversal) from **acquiescence** (swing, the part
that flips with framing).

Preprint: *Mapping Geopolitical Bias in 11 Large Language Models: A Bilingual,
Dual-Framing Analysis of U.S.-China Tensions*, arXiv:2503.23688.

The instrument poses each proposition in an affirmative direction and again with the two
compared **outcome terms** swapped, scores every answer on a polarity-aligned axis
(−2 Pro-China … +2 Pro-U.S.), and decomposes each model into net bias and swing.

## Repository layout

```
data/      input data + derived tables (the analysis outputs are written here)
code/      the instrument and all analysis scripts
figures/   the figures as they appear in the paper (main + supplementary)
requirements.txt   pinned software environment
LICENSE            MIT (applies to code)
DATA_LICENSE.md    CC-BY-4.0 (applies to data)
```

## How to run

Scripts read and write files in the current working directory, so run them **from `data/`**:

```bash
pip install -r requirements.txt
cd data
python ../code/decompose_bias.py        # net bias / swing / per-model tests -> bias_decomposition.csv
python ../code/compute_topic_stats.py   # per-model, per-topic stats -> topic_model_stats.csv
python ../code/analysis_nature.py       # variance partition, language gating, PCA -> several .csv + 2 figs
python ../code/stats_robust.py          # cluster-robust OLS (cell- and model-clustered)
python ../code/analyze_human_calib.py   # human-vs-judge calibration
python ../code/make_figures.py          # main figures (origin, language, heatmap, variance)
python ../code/make_topic_figures.py    # per-topic panels (Supp. Figs. S5-S11)
```

(Regenerated outputs land in `data/`; the published copies are in `figures/`.)

## Data (`data/`)

| File | Contents |
|---|---|
| `geobias_report_64.xlsx` | **Main crossing: 19,712 scored responses** (11 models × 7 topics × 2 languages × 2 framings × 64 iterations), with raw text, category, score, and refusal/neutral flags. |
| `judge_validation_3way.xlsx`, `judge_validation_cn.xlsx` | 1,505 free-text responses scored by judges of different origin (U.S. / Chinese). |
| `pilot_transfer.xlsx` | 2,816 responses in two non-geopolitical transfer domains. |
| `pilot_lang_pt.xlsx` | 4,928 Portuguese third-language control responses. |
| `calibration_sheet.xlsx`, `calibration_key.xlsx`, `calibration_human_labels.xlsx`, `human_calibration_merged.csv` | Human calibration of the judge (150 disagreement-enriched items). |
| `*.csv` (e.g. `bias_decomposition`, `topic_model_stats`, `variance_decomposition`, `reversal_test`, `language_gating`, `pca_coordinates`, `topic_origin_effect`, `acquiescence`, `pilot_transfer_stats`, `lang_pt_comparison`, `judge_validation_*`) | Derived tables produced by the analysis scripts (provided so results can be checked without re-running). |

## Code (`code/`)

| File | Purpose |
|---|---|
| `geobias_app.py` | **The instrument.** Stimuli (`TOPICS`), per-topic polarity keys (`AFFIRMATIVE_POLE`), wrapper pools (`PREFIXES`, `SUFFIXES`), the judge prompt, the model list, the query + scoring pipeline, and `make_topic_figure()`. Run it to collect new main-crossing data (needs an API key, see below). |
| `pilot_transfer.py`, `pilot_lang_pt.py` | Collect the transfer-domain and Portuguese-control responses (need an API key). |
| `decompose_bias.py` | Net bias, swing, raw agreement, per-model tests against zero (28-cell and 14-cell units, Holm). |
| `compute_topic_stats.py` | Per-model, per-topic polarity-aligned statistics. |
| `analysis_nature.py` | Variance partition, language gating, PCA, and the topic/acquiescence summaries → several `.csv` plus `fig_language_gating.png` and `fig_ideological_map.png`. |
| `stats_robust.py` | Variance partition and cluster-robust OLS (cell- and model-clustered). |
| `analyze_human_calib.py` | Human-vs-judge calibration (exact agreement, weighted kappa, direction-flip rate). |
| `analyze_reversal.py`, `analyze_lang_pt.py`, `analyze_pilot.py` | Reversal regimes, Portuguese control, transfer-domain analyses. |
| `build_validation_artifacts.py` | Judge-validation tables. |
| `make_figures.py`, `make_topic_figures.py`, `fig5_swing_vs_bias.py`, `fig_S_transfer.py`, `fig_overview.py` | Figure generation. |

Note on figures: every analytical figure is reproducible from the scripts above.
Main-text Fig. 5 (`figures/FIGS.png`) is a composite of the per-topic panels produced by
`make_topic_figures.py`. Several scripts `import geobias_app`, which pulls in `gradio`
for its stimulus/wrapper constants; it is listed in `requirements.txt`.

## Collecting new data (running the instrument)

The collection scripts query models through the OpenRouter API:

```bash
export OPENROUTER_API_KEY=...   # required
export OPENROUTER_PROXY=...     # optional proxy URL; defaults to none
```

No API key is stored in this repository.

## Licence

- **Code:** MIT (`LICENSE`).
- **Data:** Creative Commons Attribution 4.0 (CC-BY-4.0) (`DATA_LICENSE.md`).
