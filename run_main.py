"""Main experiment: train Mamba, Transformer and the hybrid in each walk-forward window and
target, then run the forecast-comparison tests.

Examples
--------
Confirmatory runs (Tables 1-3, 5-7, Table A):
    python run_main.py --seeds 3 4 5
Exploratory runs and the lambda sensitivity analysis (Tables 4 and 9):
    python run_main.py --seeds 0 1 2 --models Mamba Transformer Hybrid \
        Hybrid_l0.001 Hybrid_l0.003 Hybrid_l0.01 --out results/exploratory

Forecasts are cached in <out>/preds, so an interrupted run resumes where it stopped and
``--no-train`` recomputes the statistics from the cached forecasts only.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mtr import config as C
from mtr.data import load_data, alignment_check, prepare_window
from mtr.training import train_eval, numpy_baselines
from mtr.stats import (dm_test, ni_pvalue, holm, make_regimes, regime_tests, pool_regimes, flops_table)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=C.CSV_FILE)
    ap.add_argument("--out", default=str(C.OUT))
    ap.add_argument("--seeds", type=int, nargs="+", default=C.CONFIRMATORY_SEEDS)
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--targets", nargs="+", default=C.TARGETS)
    ap.add_argument("--no-train", action="store_true", help="use cached forecasts only")
    return ap.parse_args()


def main():
    args = parse_args()
    C.SEEDS, C.MODELS, C.TARGETS, C.OUT = args.seeds, args.models, args.targets, Path(args.out)
    (C.OUT / "preds").mkdir(parents=True, exist_ok=True)
    print(f"Device: {C.DEVICE} | seeds {C.SEEDS} | models {C.MODELS}")
    t0 = time.time()

    df, feat, TG, info = load_data(args.csv)
    alignment_check(info).to_csv(C.OUT / "0_alignment.csv")
    featv_raw = feat.values.astype(float)
    print(f"Rows: {len(df):,} | features: {feat.shape[1]} | {df[C.DATE_COL].iloc[0]} to {df[C.DATE_COL].iloc[-1]}")

    overall, dm_rows, reg_all, seed_rows, core_L = [], [], [], [], {}
    for target in C.TARGETS:
        for w in range(len(C.WINDOWS)):
            W = prepare_window(featv_raw, TG, info, target, w)
            if W is None:
                continue
            H, te, va, tr_fit, y_std = W["H"], W["te"], W["va"], W["tr_fit"], W["y_std"]
            print(f"\n=== target={target} | window {w}: test {W['ts']}..{W['te_end']} | train {len(tr_fit):,} "
                  f"(stride {C.TRAIN_STRIDE}) | val {len(va):,} | test {len(te):,} ===")

            preds = numpy_baselines(W["featv"], info, y_std, tr_fit, te, target)
            hyb, use_by_model = None, {}
            for seed in C.SEEDS:
                for m in C.MODELS:
                    f = C.OUT / "preds" / f"{target}_w{w}_{m}_s{seed}.npz"
                    if f.exists():
                        o = dict(np.load(f))
                    elif not args.no_train:
                        t1 = time.time()
                        o = train_eval(m, W["Fg"], y_std, tr_fit, va, te, seed)
                        np.savez(f, **{k: np.asarray(v) for k, v in o.items()})
                        print(f"  {m:<14} seed {seed} | {time.time() - t1:5.0f}s | "
                              f"MSE {np.mean((o['pred'] - y_std[te]) ** 2):.5f} | Transformer {100 * np.mean(o['use']):.1f}%")
                    else:
                        continue
                    preds[m if len(C.SEEDS) == 1 else f"{m}_s{seed}"] = o["pred"]
                    if m == "Hybrid" and seed == C.SEEDS[0]:
                        hyb = o
                    use_by_model.setdefault(m, []).append(float(np.mean(o["use"])))
                    seed_rows.append(dict(target=target, window=w, model=m, seed=seed,
                                          MSE=float(np.mean((o["pred"] - y_std[te]) ** 2)),
                                          trans_usage=float(np.mean(o["use"]))))
            # with several seeds, each neural model is represented by its seed-averaged forecast
            if len(C.SEEDS) > 1:
                for m in C.MODELS:
                    ks = [k for k in preds if k.startswith(m + "_s")]
                    if ks:
                        preds[m] = np.mean([preds[k] for k in ks], 0)

            yt = y_std[te]
            L = {k: (v - yt) ** 2 for k, v in preds.items() if "_s" not in k}
            base = L["Mean"].mean()
            for k, l in L.items():
                overall.append(dict(target=target, window=w, model=k, MSE=l.mean(), R2_vs_mean=1 - l.mean() / base,
                                    trans_usage=float(np.mean(use_by_model[k])) if k in use_by_model else np.nan))
            core_L.setdefault(target, []).append(L)

            hyb_names = [m for m in C.MODELS if m.startswith("Hybrid")]
            cand = [("Mamba", "Transformer")] + [(h_, b_) for h_ in hyb_names for b_ in ("Mamba", "Transformer")] + \
                   [("Hybrid", "Ridge"), ("Hybrid", "HAR"), ("Mamba", "HAR"), ("Transformer", "HAR")]
            pairs = [(a, b) for a, b in cand if a in L and b in L]
            for a, b in pairs:
                md, st, p = dm_test(L[a], L[b], H)
                dm_rows.append(dict(target=target, window=w, A=a, B=b, diff=md, DM=st, p=p))

            neural = {"Mamba", "Transformer", *hyb_names}
            reg_pairs = [pr for pr in pairs if pr[0] in neural and pr[1] in neural]
            reg = make_regimes(info, te, W["train_rows"], hyb["prob"] if hyb is not None else None)
            rt = regime_tests(L, reg, reg_pairs, H)
            rt.insert(0, "window", w); rt.insert(0, "target", target)
            reg_all.append(rt)
            print(f"  (elapsed: {(time.time() - t0) / 60:.1f} min)")

    # ---------------------------------------------------------------- full out-of-sample period
    ov, dm, sd_tab = pd.DataFrame(overall), pd.DataFrame(dm_rows), pd.DataFrame(seed_rows)
    rg = pd.concat(reg_all, ignore_index=True) if reg_all else pd.DataFrame()
    core_rows, pareto_rows = [], []
    for tg_, Ls in core_L.items():
        common = set.intersection(*[set(l) for l in Ls])
        Lc = {k: np.concatenate([l[k] for l in Ls]) for k in common}     # windows concatenated in time
        H = C.TARGET_H[tg_]
        hybs = sorted(k for k in common if k.startswith("Hybrid"))
        fam = [("Mamba", "Transformer")] + [(h_, b_) for h_ in hybs for b_ in ("Mamba", "Transformer")]
        for a, b in [pr for pr in fam if pr[0] in Lc and pr[1] in Lc]:
            md, st, p = dm_test(Lc[a], Lc[b], H)
            core_rows.append(dict(target=tg_, A=a, B=b, rel_diff_pct=100 * md / Lc[b].mean(), DM=st, p=p,
                                  p_noninferior=ni_pvalue(Lc[a], Lc[b], H, C.NI_MARGIN)))
        for k in sorted(common):
            if k == "Mean":
                continue
            u = sd_tab[(sd_tab.target == tg_) & (sd_tab.model == k)]["trans_usage"].mean() \
                if (k in C.MODELS and len(sd_tab)) else np.nan
            pareto_rows.append(dict(target=tg_, model=k, MSE_all_OOS=Lc[k].mean(), trans_usage=u))
    core = pd.DataFrame(core_rows)
    if len(core):
        core["p_holm"] = core.groupby("target")["p"].transform(holm)
    pareto = pd.DataFrame(pareto_rows)
    pooled = pd.DataFrame()
    if len(rg):
        rg["p_holm_in_window"] = rg.groupby(["target", "window"])["p"].transform(holm)
        pooled = pool_regimes(rg)
    if len(dm):
        dm["p_holm"] = dm.groupby("target")["p"].transform(holm)
    fl = flops_table()

    for name, t in [("1_mse_by_window", ov), ("2_dm_by_window", dm), ("3_regimes_by_window", rg),
                    ("4_regimes_pooled", pooled), ("5_per_seed", sd_tab), ("6_core_claims", core),
                    ("7_mse_full_oos_and_usage", pareto), ("8_flops", fl)]:
        t.to_csv(C.OUT / f"{name}.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    if ov.empty:
        print("\nNo results: no window had enough data. Check the date range of the CSV.")
        return
    print("\n=== MSE (standardised target) by window ===")
    print(ov.pivot_table(index=["target", "window"], columns="model", values="MSE").round(5).to_string())
    print(f"\n=== Core claims, full out-of-sample period (Holm within target; non-inferiority margin "
          f"{100 * C.NI_MARGIN:.0f}%) ===")
    print(core.round(4).to_string(index=False))
    print("\n=== MSE over the full out-of-sample period vs Transformer usage ===")
    print(pareto.sort_values(["target", "MSE_all_OOS"]).round(5).to_string(index=False))
    if len(C.SEEDS) > 1 and len(sd_tab):
        print("\n=== Variation across seeds ===")
        print(sd_tab.groupby(["target", "window", "model"]).agg(
            MSE_mean=("MSE", "mean"), MSE_sd=("MSE", "std"),
            use_min=("trans_usage", "min"), use_max=("trans_usage", "max")).round(5).to_string())
    print("\n=== Theoretical forward cost per input (MFLOP) ===")
    print(fl.round(2).to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. CSV files in {C.OUT}/")


if __name__ == "__main__":
    main()
