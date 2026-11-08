"""Figures 1-3 of the paper from the cached forecasts and tables
(run run_main.py and run_extended.py with the confirmatory seeds first).

    python make_figures.py --seeds 3 4 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mtr import config as C
from mtr.data import load_data, prepare_window
from mtr.training import numpy_baselines


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=C.CSV_FILE)
    ap.add_argument("--out", default=str(C.OUT))
    ap.add_argument("--seeds", type=int, nargs="+", default=C.CONFIRMATORY_SEEDS)
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def save(fig, path, dpi):
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    fig.savefig(path.with_suffix(".tif"), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def load_preds(target, w, m, seeds):
    return [np.load(C.OUT / "preds" / f"{target}_w{w}_{m}_s{s}.npz") for s in seeds]


def main():
    args = parse_args()
    C.OUT = Path(args.out)
    FIG = C.OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    df, feat, TG, info = load_data(args.csv)
    featv_raw = feat.values.astype(float)
    d_all = pd.to_datetime(info["dates"])
    close = info["close"]

    # ------------------------------------------------ Fig 1: routing decisions over time (grayscale)
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
    te_first = None
    for i, target in enumerate(C.TARGETS):
        tt, sel = [], []
        for w in range(len(C.WINDOWS)):
            W = prepare_window(featv_raw, TG, info, target, w)
            if W is None:
                continue
            hs = load_preds(target, w, "Hybrid", args.seeds)
            tt.append(W["te"])
            sel.append(np.mean([o["prob"] > 0.5 for o in hs], 0))
        tt, sel = np.concatenate(tt), np.concatenate(sel)
        if te_first is None:
            te_first = tt
        roll = pd.Series(sel, index=d_all[tt + 1]).rolling("7D").mean()
        ax[i + 1].fill_between(roll.index, 0, 100 * roll.values, step="post", color="0.75")
        ax[i + 1].set_ylim(0, 102)
        ax[i + 1].set_ylabel("Transformer\nusage [%]")
        ax[i + 1].set_title(f"target {target}: 7-day share of inputs routed to the Transformer "
                            f"(mean over seeds {args.seeds})", fontsize=8, loc="left")
    ax[0].plot(d_all[te_first + 1], close[te_first + 1], lw=0.6, color="k")
    ax[0].set_ylabel("GBP/USD close")
    ax[0].set_title("GBP/USD hourly close price", fontsize=8, loc="left")
    for a in ax:
        for ts, _ in C.WINDOWS[1:]:
            a.axvline(pd.Timestamp(ts), color="0.35", ls="--", lw=0.7)
    fig.tight_layout()
    save(fig, FIG / "Fig1", args.dpi)

    # ------------------------------------------------ Fig 2: forecasts in the 2022 gilt crisis (y_rv24)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10})
    target, w = "rv24", 1                                   # window 2022/23
    W = prepare_window(featv_raw, TG, info, target, w)
    te, sd, mu = W["te"], W["sd"], W["mu"]
    P = {m: np.mean([o["pred"] for o in load_preds(target, w, m, args.seeds)], 0)
         for m in ["Hybrid", "Mamba", "Transformer"]}
    P["HAR"] = numpy_baselines(W["featv"], info, W["y_std"], W["tr_fit"], te, target)["HAR"]
    P["Realised"] = W["y_std"][te]
    dd = pd.DatetimeIndex(d_all[te])
    m = (dd >= pd.Timestamp(C.FORECAST_PLOT[0])) & (dd < pd.Timestamp(C.FORECAST_PLOT[1]))
    s = pd.DataFrame({k: v[m] * sd + mu for k, v in P.items()}, index=dd[m])
    s = s[~s.index.duplicated()].resample("1h").asfreq()   # weekends shown as gaps
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(s.index, s["Realised"], color="black", lw=1.5, label="Realised (next 24 h)")
    sty = {"Hybrid": dict(color="#1f77b4", lw=1.5),
           "Transformer": dict(color="#2ca02c", lw=0.9, ls="--"),
           "Mamba": dict(color="#ff7f0e", lw=0.9, ls=":"),
           "HAR": dict(color="#d62728", lw=0.9, ls="-.")}
    for k, st in sty.items():
        ax.plot(s.index, s[k], label=k, **st)
    ev = pd.Timestamp("2022-09-23")
    ax.axvline(ev, color="grey", lw=0.8, ls="--")
    ax.text(ev, ax.get_ylim()[1], " mini-budget (23 Sep)", va="top", fontsize=9, color="grey")
    ax.set_ylabel("log realised volatility (24 h)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.grid(alpha=0.3)
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    save(fig, FIG / "Fig2", args.dpi)

    # ------------------------------------------------ Fig 3: regime-wise Mamba vs Transformer
    lab = {"vol_low": "Low volatility", "vol_mid": "Medium volatility", "vol_high": "High volatility",
           "atr_top10": "ATR top decile (1 window)", "sess_asia_00-07": "Asian session 00–07",
           "sess_london_07-12": "London 07–12", "sess_overlap_12-16": "London–NY overlap 12–16",
           "sess_ny_16-21": "New York 16–21", "sess_late_21-24": "Late trading 21–24",
           "mini_budget_2022": "Mini-budget 2022 (1 window)"}
    pooled = pd.read_csv(C.OUT / "extended" / "H_regimes_pooled.csv")
    pm = pooled[(pooled.A == "Mamba") & (pooled.B == "Transformer") & pooled.regime.isin(list(lab))]
    order = list(lab)[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for a, (tg, name) in zip(axs, [("abs", "1-hour horizon (y_abs)"), ("rv24", "24-hour horizon (y_rv24)")]):
        q = pm[pm.target == tg].set_index("regime").reindex(order)
        rob = q["robust"].astype(object).fillna(False).astype(bool)
        a.barh([lab[r] for r in order], q["mean_rel_pct"], color=["#c0392b" if r else "#9e9e9e" for r in rob])
        a.axvline(0, color="k", lw=0.8)
        a.grid(axis="x", alpha=0.3)
        a.set_title(name, fontsize=11)
        a.set_xlabel("MSE(Mamba) / MSE(Transformer) − 1  [%]\n← Mamba more accurate  |  Transformer more accurate →",
                     fontsize=9)
    fig.tight_layout()
    save(fig, FIG / "Fig3", args.dpi)
    print(f"Figures saved in {FIG}/")


if __name__ == "__main__":
    main()
