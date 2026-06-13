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

All files are at the top level; analysis scripts read the data files from the working
directory.

### Data
| File | Contents |
|---|---|
| `geobias_report_64.xlsx` | **Main crossing: 19,712 scored responses** (11 models × 7 topics × 2 languages × 2 framings × 64 iterations). Columns include `Model, Language, Framing, Iteration, Topic, Raw, Score, IsRefusal, IsNeutral`. |
| `judge_validation_3way.xlsx` | 1,505 free-text responses scored by judges of different origin (U.S. / Chinese). |
| `judge_validation_cn.xlsx` | Chinese-judge re-scores used for the judge-robustness check. |
| `pilot_transfer.xlsx` | 2,816 responses in two non-geopolitical transfer domains (cultural values, scientific consensus). |
| `pilot_lang_pt.xlsx` | 4,928 Portuguese third-language control responses. |
| `calibration_sheet.xlsx`, `calibration_key.xlsx`, `calibration_human_labels.xlsx`, `human_calibration_merged.csv` | Human calibration of the judge (150 disagreement-enriched items). |
| `bias_decomposition.csv`, `topic_model_stats.csv`, `variance_decomposition.csv`, `reversal_test.csv`, `language_gating.csv`, `pca_coordinates.csv`, `topic_origin_effect.csv`, `acquiescence.csv`, `lang_pt_comparison.csv`, `pilot_transfer_stats.csv`, `judge_validation_*.csv`, `pool_split.csv` | Derived tables produced by the analysis scripts (provided so results can be checked without re-running). |

### Code
| File | Purpose |
|---|---|
| `geobias_app.py` | **The instrument.** Stimuli (`TOPICS`), per-topic polarity keys (`AFFIRMATIVE_POLE`), wrapper pools (`PREFIXES`, `SUFFIXES`), the judge prompt, the model list, and the query + scoring pipeline. Run it to collect new data (requires an API key, see below). |
| `decompose_bias.py` | Net bias, swing, raw agreement, and per-model tests against zero (28-cell and 14-cell units, Holm) → `bias_decomposition.csv`. |
| `compute_topic_stats.py` | Per-model, per-topic polarity-aligned statistics → `topic_model_stats.csv`. |
| `stats_robust.py` | Variance partition and cluster-robust OLS (cell- and model-clustered). |
| `make_figures.py`, `fig5_swing_vs_bias.py`, `fig_S_transfer.py`, `fig_overview.py` | Figure generation. |
| `analyze_human_calib.py` | Human-vs-judge calibration (exact agreement, weighted kappa, direction-flip rate). |
| `analyze_reversal.py`, `analyze_lang_pt.py`, `analyze_pilot.py` | Reversal regimes, Portuguese control, and transfer-domain analyses. |
| `build_validation_artifacts.py` | Judge-validation tables. |
| `requirements.txt` | Pinned environment (`scipy` is pinned because `wilcoxon`'s tie handling varies across versions). |

## Reproduce

```bash
pip install -r requirements.txt
python decompose_bias.py        # net bias / swing / per-model tests
python compute_topic_stats.py   # per-topic stats
python stats_robust.py          # variance partition + cluster-robust OLS
python analyze_human_calib.py   # human-vs-judge calibration
python make_figures.py          # figures
```

Note: `make_figures.py`, `compute_topic_stats.py`, and `build_validation_artifacts.py`
`import geobias_app`, which pulls in `gradio` for its stimulus/wrapper constants; it is
included in `requirements.txt`.

## Collecting new data (running the instrument)

`geobias_app.py` queries models through the OpenRouter API. Set:

```bash
export OPENROUTER_API_KEY=...        # required
export OPENROUTER_PROXY=...          # optional (e.g. a VPN/proxy URL); defaults to none
```

No API key is stored in this repository.

## Licence

- **Code:** MIT (see `LICENSE`).
- **Data:** Creative Commons Attribution 4.0 (CC-BY-4.0) (see `DATA_LICENSE.md`).
