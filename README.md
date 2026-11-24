# Can machine learning nowcast core CPI better than the forecasters? A random forest, XGBoost and a linear model against the Reuters poll, 2021–2026

Every month, the day before the U.S. consumer price index (CPI) comes out, economists polled by Reuters publish their forecast of how much core CPI (all items except food and energy) rose that month. Markets trade on the gap between that forecast and the print. This repository asks whether machine-learning models, fed only information that was public the day before each release, can nowcast the monthly change in core CPI better than the poll, and in particular whether they could have caught the jump in 2021, when the poll missed by more than at any other time since 2004.

Three models (a random forest, gradient boosting with XGBoost, and their equal-weight average) and a linear regression are compared with three benchmarks: last month's print (persistence), the average of the last 12 prints, and the Reuters poll median. Everything was chosen on data before 2021; the test is January 2021 to October 2026 (68 monthly nowcasts).

Findings:

- **Error over the whole test (root mean squared error, percentage points):** Reuters poll 0.149, linear regression 0.147, random forest 0.189, XGBoost 0.186, the average of the two tree models 0.187, last month's print 0.176, 12-print average 0.180.
- **The tree models lost to the poll, and the gap is unlikely to be luck** (Diebold–Mariano test of equal squared error, p = 0.015 for the random forest, 0.019 for XGBoost, 0.018 for their average). They did no better than "same as last month" either (p ≈ 0.6–0.7).
- **The linear regression tied the poll** (0.147 against 0.149; p = 0.83). It was the only method closer than the poll in 2021 (0.238 against 0.268).
- **April 2021**, the poll's biggest miss: core printed 0.9; the poll said 0.3; the linear regression 0.37; both tree models 0.18. Before 2021, 98.9% of first prints were 0.3 or below and only one was above 0.5 (July 2020). A tree's nowcast is an average of training months, so it cannot go far beyond what it has seen; a linear model can.
- **What moved the nowcasts (SHAP):** core CPI's own past prints, for every model. In the linear regression, used-car prices added 0.15 points to the April 2021 nowcast; jet fuel added under 0.01 points to an average tree nowcast.
- **Cross-validation could barely tell the input sets apart.** For each model, the best version of every input set scored within 0.001 points of the others before 2021, and which set won depended on whether 2019–20 (the pandemic) was in the scoring.

Published post: https://1iiiiii.github.io/Personal-Website/blog/posts/post10/

## Data

| File | What it contains | Source | Period |
|---|---|---|---|
| `data/raw/core_cpi_lseg_first_release_poll.csv` | for each month: the change in core CPI (all items less food and energy, seasonally adjusted, %) **as first published**, the release date, and the median of the Reuters poll of forecasters | LSEG Workspace, *United States Core CPI MM, SA* (`USCPF=ECI`, poll median `pUSCPF=M`). The table was captured from LSEG's Indicator Explorer on 20 Nov 2026 and typed in, because the Excel add-in is disabled and the Python API needs a separate licence; rows captured twice matched exactly | poll from Jun 1999, first prints from Apr 2004, to Oct 2026 |
| `data/raw/manheim_used_vehicle_value_index.xlsx` | Manheim Used Vehicle Value Index: wholesale used-car prices, seasonally adjusted, Jan 1997 = 100 | Cox Automotive data file, downloaded 21 Nov 2026 | Jan 1997 – Oct 2026 |
| `data/raw/fred_jet_fuel_monthly.csv` | U.S. Gulf Coast kerosene-type jet fuel, $ per gallon, monthly average of daily prices | FRED `DJFUELUSGULF` (monthly average computed by FRED), downloaded 21 Nov 2026 | Feb 1999 – Oct 2026 |
| `data/raw/fred_cpi_used_cars.csv` | CPI used cars and trucks, seasonally adjusted | FRED `CUSR0000SETA02`, downloaded 21 Nov 2026 | Jan 1997 – Oct 2026 |
| `data/raw/fetch_log.json` | how and when each file was obtained, with the FRED URLs and file checksums | | |

October 2025 was never published (the BLS did not collect October prices during the government shutdown) and November 2025 was published only as a two-month change, so neither has a one-month first print. Both drop out as targets.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13. `requirements.txt` pins the exact package versions the results were produced with; with those, every table reproduces exactly. The code also runs with older versions (tested with pandas 2.3, scikit-learn 1.7, XGBoost 3.2, SHAP 0.49): the choices made by cross-validation are the same and the scores differ only in the fourth decimal.

Open the notebooks in VS Code with the `.venv` kernel selected. No API keys are needed. Every random draw starts from a fixed seed (6400). To download the two FRED files again, set `REFETCH = True` in notebook 01; the LSEG and Manheim files are manual downloads.

## Replication

Run the notebooks in `code/` in order; each reads the previous ones' output. Paths are built from `Path.cwd().parent`, so run them from `code/` (VS Code's default). Shared settings (inputs, folds, hyperparameter grids, seeds) live in `code/nowcast_setup.py`.

| Notebook | Reads | Writes |
|---|---|---|
| `01_load.ipynb` | the four raw files | `data/processed/monthly.csv`; `results/tables/data_coverage.csv` |
| `02_features.ipynb` | `monthly.csv` | `data/processed/features.csv`; `results/tables/design_checks_lead.csv`, `naive_benchmarks_before_2021.csv`, `timing_check.json` |
| `03_tune.ipynb` | `features.csv`, months before 2021 only | `results/tables/cv_grid.csv`, `cv_best_by_feature_set.csv`, `cv_benchmarks.csv`, `cv_choice.json` |
| `04_test.ipynb` | `features.csv`, `cv_choice.json`, `cv_best_by_feature_set.csv` | `data/processed/nowcasts.csv`; `results/tables/test_metrics.csv`, `test_rmse_by_year.csv`, `test_biggest_poll_misses.csv`, `test_feature_sets_not_used_for_choice.csv` |
| `05_interpret.ipynb` | `features.csv`, `nowcasts.csv`, `cv_choice.json` | `results/tables/shap_values.csv`, `shap_importance.csv`, `shap_block_importance.csv`, `shap_2021.csv`, `ols_coefficients_before_2021.csv` |
| `06_figures.ipynb` | `nowcasts.csv`, `test_metrics.csv`, `shap_values.csv` | `results/figures/fig1_nowcasts.html`, `fig2_errors.html`, `fig3_shap.html` (also copied into the post folder when the website repository sits next to this one) |

From the terminal:

```bash
cd code
for n in 01_load 02_features 03_tune 04_test 05_interpret 06_figures; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $n.ipynb
done
```

Notebooks 03 and 04 take about two and a half minutes each; the rest a few seconds. A full re-run from an empty `data/processed/` and `results/` reproduced every table byte for byte, and a second run from a fresh copy with older package versions gave the same choices.

## Method

**The nowcast day.** Each nowcast is made the day before that month's CPI release, which is roughly when the Reuters poll is collected, so the models and the poll work from the same information. An input is used only if it was published before that day (checked in notebook 02).

**Target.** The first print of core CPI's monthly change, seasonally adjusted, in percent: the number the market reacted to. Not the year-over-year rate, because 11 of its 12 monthly changes are already public on the nowcast day; not the unadjusted change, because a model could then "beat" persistence just by learning that prices jump every January.

**Inputs** (8 candidates):

| Input | What it is | Why |
|---|---|---|
| `core_last`, `core_avg3`, `core_avg12` | the last first print, and the averages of the last 3 and 12 | core inflation is persistent; the three cover the latest reading, short-term momentum and the trend |
| `manheim_0`, `_1`, `_2` | % change in Manheim's wholesale used-car index in the same month and the two before | used cars are about 3% of core but swing several percent in a month; in April 2021 they added about 0.35 points to core. Manheim's full-month index comes out on the 5th business day of the next month, before CPI |
| `jet_0`, `_1` | % change in the monthly average jet fuel price, same month and one before | jet fuel feeds airfares, another small but volatile part of core |

Wages, producer prices and shelter leads were considered and left out: wages move too smoothly to explain monthly swings, producer prices come out before CPI only in some months, and market rents lead CPI shelter by many months (their trend is already in the 12-print average). The lags were set from correlations on 2005–2020 only (`design_checks_lead.csv`): Manheim leads the CPI's own used-car index by two to three months, but its link with core CPI is strongest in the same month and the next.

**Split.** In time order. Test: January 2021 – October 2026, untouched until notebook 04. Before that, expanding-window cross-validation with four 24-month validation blocks (2013–14, 2015–16, 2017–18, 2019–20), each scored by a model trained on every month before it from April 2005. The score is the average of the four blocks' RMSEs, which keeps 2020 to one block of four.

**What cross-validation chose**, separately for each model: the hyperparameters, and one of four input sets (core history alone, plus used cars, plus jet fuel, plus both). An input set had to lower the cross-validated error to be chosen; training error cannot judge an input, because any extra column, even noise, lowers it.

| Model | Grid | Chosen | CV RMSE |
|---|---|---|---|
| Linear regression | – | core + used cars | 0.101 |
| Random forest (300 trees) | depth 2, 3, 5, unlimited × min. leaf 5, 10, 20 × share of inputs per split 1.0, 0.5 | core + jet fuel; depth 2, min. leaf 5, half the inputs per split | 0.105 |
| XGBoost (row and column subsampling 0.8) | trees 100, 300, 600 × learning rate 0.01, 0.05 × depth 1, 2, 3 × min. child weight 5, 10 | core + jet fuel; 300 trees, rate 0.01, depth 1, min. child weight 5 | 0.104 |
| Benchmarks on the same blocks | | poll 0.097, 12-print average 0.110, last print 0.125 | |

Without the 2019–20 block the winning input set changes for all three models (core alone for the linear regression and the random forest; all inputs for XGBoost): the input sets are close to a tie.

**Test.** Settings fixed; each month every model is refitted on all months whose first print was out before that nowcast day, then makes one nowcast. Scores: RMSE overall, in 2021 and in 2022–2026; MAE; RMSE with nowcasts rounded to 0.1 like the published numbers; share of months closer than the poll; the Diebold–Mariano test (Harvey–Leybourne–Newbold version) against the poll and against persistence.

**Interpretation.** SHAP values for every test nowcast, computed on the model that made it: `shap.TreeExplainer` for the random forest, XGBoost's own TreeSHAP for XGBoost, coefficient × distance from the training mean for the linear regression.

## Results (`results/tables/`)

Out of sample, January 2021 – October 2026, 68 months. RMSE and MAE in percentage points.

| Method | RMSE | RMSE 2021 | RMSE 2022–26 | MAE | RMSE, rounded to 0.1 | Closer than the poll | DM p vs poll | DM p vs persistence |
|---|---|---|---|---|---|---|---|---|
| Last month's print (persistence) | 0.176 | 0.301 | 0.134 | 0.124 | 0.176 | 22% | 0.205 | |
| Average of the last 12 prints | 0.180 | 0.344 | 0.119 | 0.124 | 0.185 | 38% | 0.030 | 0.855 |
| Reuters poll (median) | 0.149 | 0.268 | 0.108 | 0.102 | 0.149 | | | 0.205 |
| Linear regression | 0.147 | 0.238 | 0.119 | 0.106 | 0.145 | 43% | 0.826 | 0.166 |
| Random forest | 0.189 | 0.368 | 0.120 | 0.128 | 0.186 | 37% | 0.015 | 0.624 |
| Gradient boosting (XGBoost) | 0.186 | 0.361 | 0.119 | 0.124 | 0.186 | 35% | 0.019 | 0.691 |
| Average of random forest and XGBoost | 0.187 | 0.364 | 0.119 | 0.125 | 0.188 | 35% | 0.018 | 0.672 |

The poll's largest misses in the test (`test_biggest_poll_misses.csv`):

| Month | First print | Poll | Linear regression | Random forest | XGBoost |
|---|---|---|---|---|---|
| Apr 2021 | 0.9 | 0.3 | 0.37 | 0.18 | 0.18 |
| Jun 2021 | 0.9 | 0.4 | 0.39 | 0.21 | 0.22 |
| Aug 2022 | 0.6 | 0.3 | 0.28 | 0.37 | 0.35 |
| May 2021 | 0.7 | 0.4 | 0.62 | 0.18 | 0.19 |

What moved the nowcasts (`shap_block_importance.csv`, average absolute contribution over the test months): core CPI history 0.134 points in the linear regression, 0.097 in the random forest, 0.108 in XGBoost; used-car prices 0.053 in the linear regression; jet fuel 0.006–0.007 in the tree models. In April 2021 the linear regression started from 0.155 (its average nowcast) and added 0.06 for last month's print and 0.15 for used-car prices (0.10 same month, 0.04 one month earlier, 0.01 two months earlier).

**Not used for any choice** (`test_feature_sets_not_used_for_choice.csv`): with hindsight, adding used-car prices to the tree models would have cut their 2021 error (random forest 0.368 → 0.298) but raised it in 2022–2026 (0.120 → 0.142), when used-car prices fell in 16 of 24 months while core stayed near 0.4.

## Caveats

- The Manheim file is Cox's latest, restated series (rebased January 2023), not the values as first published each month. Seasonal factors and revisions since then are small next to the 2021 moves, but the used-car inputs are not strictly real-time.
- The core prints and poll were transcribed from screenshots; rows captured twice matched exactly, but there is no machine export to check against.
- The first print and the poll are rounded to 0.1 point; model nowcasts are not. The rounded-nowcast column shows this changes little.
- 68 test months, 12 of them in 2021. The Diebold–Mariano p-values assume the loss differences are independent from month to month.
- The input sets were nearly tied in cross-validation, so the choice between them carries little information.

## Layout

```
code/             01_load … 06_figures (notebooks), nowcast_setup.py (shared settings), plotly_style.py
data/raw/         the four source files and fetch_log.json
data/processed/   monthly.csv, features.csv, nowcasts.csv
results/tables/   every number quoted in the post
results/figures/  the three interactive HTML figures
```
