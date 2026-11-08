# Task-adaptive hard routing between Mamba and Transformer experts

Code accompanying the paper

> Kęsicki D, Bernardelli M. *Task-adaptive hard routing between Mamba and Transformer experts for financial volatility forecasting.*

A learnable router, trained with the straight-through Gumbel-Softmax estimator, sends each input
window either to a selective state-space (Mamba) expert or to a Transformer expert. At inference only
the selected expert is executed. The experiments forecast two volatility targets of hourly GBP/USD
data (next-hour absolute return and 24-hour-ahead log realised volatility) in a three-window
walk-forward design, with Diebold-Mariano, non-inferiority and Holm-corrected tests.

## Repository layout

| File | Purpose |
|---|---|
| `mtr/config.py` | all settings reported in the paper (Section 5, Table 10) |
| `mtr/data.py` | loading, stationary feature transforms, targets, walk-forward splits, scaling |
| `mtr/models.py` | pure-PyTorch Mamba (S6) block, Transformer block, routed hybrid, DLinear/NLinear/PatchTST |
| `mtr/training.py` | training with early stopping, hard-routed prediction, Ridge and HAR baselines |
| `mtr/stats.py` | Diebold-Mariano (HAC), non-inferiority, Holm/BH, block-bootstrap regime tests, FLOP model |
| `aggregate_histdata.py` | aggregates HistData.com one-minute bars to hourly OHLC bars |
| `build_features.py` | builds the model input file (technical indicators, calendar encodings) from hourly OHLC bars |
| `run_main.py` | main experiment: experts and hybrid, core tests, regimes, per-seed results, FLOPs |
| `run_extended.py` | baselines, validation selection, expert averaging, random router, ablation, regret |
| `make_figures.py` | Figures 1–3 (PNG and TIFF) |

The selective scan is written in plain PyTorch, so no `mamba_ssm` CUDA kernels are needed.

## Requirements

Python 3 and the packages in `requirements.txt`. The experiments in the paper were run on
Google Colab with one GPU; a single model trains in about 20 seconds there. The code also runs on CPU (slower).

```bash
pip install -r requirements.txt
```

## Data

The prices are the free one-minute GBP/USD bars from [HistData.com](https://www.histdata.com)
(2000–2024), aggregated to hourly bars. Download the "Generic ASCII" M1 files and build the input file:

```bash
python aggregate_histdata.py --input-dir raw_histdata --output data/gbpusd_1h_ohlc.csv
python build_features.py --input data/gbpusd_1h_ohlc.csv --output data/gbpusd_1h_features.csv
```

See `data/README.md` for the column definitions.

## Reproducing the results

```bash
# 1. Confirmatory seeds 3-5: Tables 1-3, 5-7 and Table A
python run_main.py --seeds 3 4 5

# 2. Extended analyses on the same seeds: baselines, random router, ablation (Tables 2, 6, 8)
python run_extended.py --seeds 3 4 5

# 3. Exploratory seeds 0-2 and the lambda sensitivity analysis (Tables 4 and 9)
python run_main.py --seeds 0 1 2 --models Mamba Transformer Hybrid \
    Hybrid_l0.001 Hybrid_l0.003 Hybrid_l0.01 --out results/exploratory

# 4. Figures 1-3
python make_figures.py --seeds 3 4 5
```

Forecasts are cached in `results/preds` (and `results/preds_extended`), so an interrupted run resumes
where it stopped; `run_main.py --no-train` recomputes all statistics from the cached forecasts.
All tables are written as CSV files to `results/` and `results/extended/`; figures to `results/figures/`.

Main outputs:

| CSV | Content |
|---|---|
| `results/1_mse_by_window.csv` | MSE of every model in each window (standardised target) |
| `results/6_core_claims.csv` | full out-of-sample DM tests, non-inferiority (2%), Holm |
| `results/7_mse_full_oos_and_usage.csv` | full out-of-sample MSE and Transformer usage (Table 9 with `Hybrid_l*`) |
| `results/5_per_seed.csv` | MSE and Transformer usage per seed |
| `results/8_flops.csv` | theoretical forward cost per input (Table A) |
| `results/extended/B_claims.csv` | hybrid vs all alternatives, non-inferiority at 0.5% and 2%, Holm |
| `results/extended/D_random_router.csv` | learned vs random routing (permutation test), anti-router, soft inference |
| `results/extended/E_ablation.csv` | router ablation |
| `results/extended/F_regret.csv` | regret relative to the better expert |
| `results/extended/H_regimes_pooled.csv` | regime analysis pooled over windows |

## Determinism

Seeds are set for Python, NumPy and PyTorch. Results are bit-for-bit reproducible on the same
hardware and software; small numerical differences can occur across GPU types or library versions.

## License

MIT (see `LICENSE`).
