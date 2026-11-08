"""Extended analyses on top of the main run (requires the cached forecasts of run_main.py
for the same seeds; missing ones are trained):

  * training-free comparisons: validation-based selection, expert averaging, random router
    with the same Transformer share (permutation test), anti-router, soft routing at inference;
  * deep-learning baselines: DLinear, NLinear, PatchTST;
  * router ablation (single seed): linear router, soft Gumbel training, no expert warm-up,
    fixed temperature 1.0 and 0.2;
  * regret relative to the better expert and regime-by-window tables.

    python run_extended.py --seeds 3 4 5
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mtr import config as C
from mtr.data import load_data, prepare_window
from mtr.models import DLinearNet, PatchTSTNet, HybridVariant
from mtr.training import train_eval, fit_eval, cached, numpy_baselines
from mtr.stats import dm_test, ni_pvalue, holm, make_regimes, regime_tests, pool_regimes, stars


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=C.CSV_FILE)
    ap.add_argument("--out", default=str(C.OUT))
    ap.add_argument("--seeds", type=int, nargs="+", default=C.CONFIRMATORY_SEEDS)
    ap.add_argument("--baseline-seeds", type=int, nargs="+", default=C.BASELINE_SEEDS)
    ap.add_argument("--ablation-seeds", type=int, nargs="+", default=C.ABL_SEEDS)
    ap.add_argument("--targets", nargs="+", default=C.TARGETS)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    C.SEEDS, C.TARGETS, C.OUT = args.seeds, args.targets, Path(args.out)
    SEEDS, BSEEDS, ASEEDS = args.seeds, args.baseline_seeds, args.ablation_seeds
    OUT, PRED, PRED_X = C.OUT / "extended", C.OUT / "preds", C.OUT / "preds_extended"
    for p in (OUT, PRED, PRED_X):
        p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, feat, TG, info = load_data(args.csv)
    featv_raw = feat.values.astype(float)

    store, abl_models, base_models = {}, [], []
    for target in C.TARGETS:
        store[target] = []
        for w in range(len(C.WINDOWS)):
            W = prepare_window(featv_raw, TG, info, target, w)
            if W is None:
                continue
            Fg, y_std, tr_fit, va, te = W["Fg"], W["y_std"], W["tr_fit"], W["va"], W["te"]
            yt, yv = y_std[te], y_std[va]
            print(f"\n=== {target} | window {w} ({W['ts']}..{W['te_end']}) | test {len(te):,} ===")

            # ---------- models of the main run (cached) ----------
            base = {}
            for m in ["Mamba", "Transformer", "Hybrid"]:
                for s in SEEDS:
                    base[(m, s)] = cached(PRED / f"{target}_w{w}_{m}_s{s}.npz",
                                          lambda m=m, s=s: train_eval(m, Fg, y_std, tr_fit, va, te, s))
            P, PV = {}, {}
            for m in ["Mamba", "Transformer", "Hybrid"]:
                P[m] = np.mean([base[(m, s)]["pred"] for s in SEEDS], 0)
                PV[m] = np.mean([base[(m, s)]["pred_va"] for s in SEEDS], 0)
            P.update(numpy_baselines(W["featv"], info, y_std, tr_fit, te, target))

            # ---------- training-free comparisons ----------
            vm = {m: np.mean((PV[m] - yv) ** 2) for m in ["Mamba", "Transformer"]}
            chosen = min(vm, key=vm.get)
            P["ValSelect"] = P[chosen]
            P["AvgMT"] = 0.5 * (P["Mamba"] + P["Transformer"])
            hs = [base[("Hybrid", s)] for s in SEEDS]
            P["Hybrid_SoftInference"] = np.mean([(1 - o["prob"]) * o["pm_int"] + o["prob"] * o["pt_int"] for o in hs], 0)
            P["Hybrid_AntiRouter"] = np.mean([np.where(o["prob"] > 0.5, o["pm_int"], o["pt_int"]) for o in hs], 0)
            rng = np.random.default_rng(0)
            rates = [np.mean(o["prob"] > 0.5) for o in hs]
            rand_L = np.empty((C.N_RANDOM_DRAWS, len(te)), np.float32)
            for r in range(C.N_RANDOM_DRAWS):
                pr = np.mean([np.where(rng.random(len(te)) < u, o["pt_int"], o["pm_int"]) for o, u in zip(hs, rates)], 0)
                rand_L[r] = (pr - yt) ** 2
                if r == 0:
                    P["Hybrid_RandomRouter"] = pr

            # ---------- deep-learning baselines ----------
            if not args.no_baselines:
                fin = W["featv"].shape[1]
                makers = {"DLinear": lambda: DLinearNet(fin, "dlinear"),
                          "NLinear": lambda: DLinearNet(fin, "nlinear"),
                          "PatchTST": lambda: PatchTSTNet(fin)}
                for m, mk in makers.items():
                    outs = [cached(PRED_X / f"{target}_w{w}_{m}_s{s}.npz",
                                   lambda mk=mk, s=s: fit_eval(mk, Fg, y_std, tr_fit, va, te, s))
                            for s in BSEEDS]
                    P[m] = np.mean([o["pred"] for o in outs], 0)
                    if m not in base_models:
                        base_models.append(m)

            # ---------- router ablation (compared at the same seeds) ----------
            use = {"Hybrid": np.mean([np.mean(base[("Hybrid", s)]["use"]) for s in SEEDS])}
            if not args.no_ablation:
                abl_base = {(m, s): cached(PRED / f"{target}_w{w}_{m}_s{s}.npz",
                                           lambda m=m, s=s: train_eval(m, Fg, y_std, tr_fit, va, te, s))
                            for m in ["Mamba", "Transformer", "Hybrid"] for s in ASEEDS}
                P["Hybrid_ref"] = np.mean([abl_base[("Hybrid", s)]["pred"] for s in ASEEDS], 0)
                use["Hybrid_ref"] = np.mean([np.mean(abl_base[("Hybrid", s)]["use"]) for s in ASEEDS])
                hs_a = [abl_base[("Hybrid", s)] for s in ASEEDS]
                P["Abl_SoftInference"] = np.mean([(1 - o["prob"]) * o["pm_int"] + o["prob"] * o["pt_int"] for o in hs_a], 0)
                use["Abl_SoftInference"] = 1.0
                for m in ["Mamba", "Transformer"]:
                    P[f"{m}_ref"] = np.mean([abl_base[(m, s)]["pred"] for s in ASEEDS], 0)
                for name, cfg in C.ROUTER_VARIANTS.items():
                    outs = [cached(PRED_X / f"{target}_w{w}_{name}_s{s}.npz",
                                   lambda cfg=cfg, s=s: fit_eval(
                                       lambda: HybridVariant(Fg.shape[1], cfg), Fg, y_std, tr_fit, va, te, s,
                                       hybrid=True, expert_warmup=cfg.get("expert_warmup", C.EXPERT_WARMUP_EPOCHS),
                                       tau_fixed=cfg.get("tau_fixed")))
                            for s in ASEEDS]
                    P[name] = np.mean([o["pred"] for o in outs], 0)
                    use[name] = np.mean([np.mean(o["use"]) for o in outs])
                    if name not in abl_models:
                        abl_models.append(name)

            store[target].append(dict(
                w=w, te=te, yt=yt, mu=W["mu"], sd=W["sd"], train_rows=W["train_rows"], chosen=chosen,
                L={k: (v - yt) ** 2 for k, v in P.items()}, P=P, rand_L=rand_L, use=use,
                sel=np.mean([o["prob"] > 0.5 for o in hs], 0),          # share of seeds routing to the Transformer
                prob0=hs[0]["prob"]))
            print(f"  validation selection chose: {chosen} | elapsed {(time.time() - t0) / 60:.1f} min")

    # ---------------------------------------------------------------- tables
    T_overall, T_claims, T_window, T_rand, T_abl, T_regret, reg_rows, T_choice = [], [], [], [], [], [], [], []
    for target, Ws in store.items():
        if not Ws:
            continue
        H = C.TARGET_H[target]
        common = set.intersection(*[set(x["L"]) for x in Ws])
        Lc = {k: np.concatenate([x["L"][k] for x in Ws]) for k in common}
        for x in Ws:
            T_choice.append(dict(target=target, window=x["w"], ValSelect_choice=x["chosen"]))
        for k in sorted(common):
            row = dict(target=target, model=k, MSE_all=Lc[k].mean())
            for x in Ws:
                row[f"MSE_w{x['w']}"] = x["L"][k].mean()
            T_overall.append(row)

        # main comparisons of the hybrid: one family per target, Holm-adjusted
        fam = [("Mamba", "Transformer"), ("Hybrid", "Mamba"), ("Hybrid", "Transformer"),
               ("Hybrid", "ValSelect"), ("Hybrid", "AvgMT"), ("Hybrid", "Ridge"), ("Hybrid", "HAR"),
               ("Hybrid", "Hybrid_RandomRouter")] + [("Hybrid", m) for m in base_models]
        rows = []
        for a, b in fam:
            if a not in Lc or b not in Lc:
                continue
            md, st, p = dm_test(Lc[a], Lc[b], H)
            r = dict(target=target, A=a, B=b, rel_diff_pct=100 * md / Lc[b].mean(), DM=st, p=p)
            for mg in C.NI_MARGINS:
                r[f"p_NI_{100 * mg:g}pct"] = ni_pvalue(Lc[a], Lc[b], H, mg)
            rows.append(r)
        rows = pd.DataFrame(rows)
        rows["p_holm"] = holm(rows["p"].values)
        T_claims.append(rows)

        for x in Ws:
            for a, b in [("Mamba", "Transformer"), ("Hybrid", "Mamba"), ("Hybrid", "Transformer"),
                         ("Hybrid", "ValSelect"), ("Hybrid", "AvgMT")]:
                md, st, p = dm_test(x["L"][a], x["L"][b], H)
                T_window.append(dict(target=target, window=x["w"], A=a, B=b,
                                     rel_diff_pct=100 * md / x["L"][b].mean(), DM=st, p=p))

        # random router: permutation test over the full out-of-sample period
        rand_all = np.concatenate([x["rand_L"] for x in Ws], 1).mean(1)
        hyb = Lc["Hybrid"].mean()
        T_rand.append(dict(target=target, MSE_hybrid=hyb, MSE_random_mean=rand_all.mean(),
                           MSE_random_q05=np.quantile(rand_all, .05), MSE_random_q95=np.quantile(rand_all, .95),
                           p_perm=(np.sum(rand_all <= hyb) + 1) / (len(rand_all) + 1),
                           MSE_anti_router=Lc["Hybrid_AntiRouter"].mean(),
                           MSE_soft_inference=Lc["Hybrid_SoftInference"].mean()))

        if "Hybrid_ref" in Lc:
            ref = Lc["Hybrid_ref"]
            for k in ["Hybrid_ref", "Mamba_ref", "Transformer_ref", "Abl_SoftInference"] + abl_models:
                if k not in Lc:
                    continue
                u = np.mean([x["use"].get(k, np.nan) for x in Ws]) if k not in ("Mamba_ref", "Transformer_ref") \
                    else (0.0 if k == "Mamba_ref" else 1.0)
                r = dict(target=target, variant=k, MSE=Lc[k].mean(), trans_usage=u,
                         delta_vs_full_pct=100 * (Lc[k].mean() / ref.mean() - 1))
                if k != "Hybrid_ref":
                    _, st, p = dm_test(Lc[k], ref, H)
                    r.update(DM_vs_full=st, p_vs_full=p)
                T_abl.append(r)

        best = min(Lc["Mamba"].mean(), Lc["Transformer"].mean())
        for k in ["Mamba", "Transformer", "Hybrid", "ValSelect", "AvgMT"]:
            T_regret.append(dict(target=target, model=k, regret_pct=100 * (Lc[k].mean() / best - 1)))

        for x in Ws:
            reg = make_regimes(info, x["te"], x["train_rows"], x["prob0"])
            rt = regime_tests(x["L"], reg, [("Mamba", "Transformer"), ("Hybrid", "Mamba"), ("Hybrid", "Transformer")], H)
            rt.insert(0, "window", x["w"]); rt.insert(0, "target", target)
            reg_rows.append(rt)

    ov, cl = pd.DataFrame(T_overall), pd.concat(T_claims, ignore_index=True)
    wn, rd, ab = pd.DataFrame(T_window), pd.DataFrame(T_rand), pd.DataFrame(T_abl)
    rgt = pd.DataFrame(T_regret).pivot(index="model", columns="target", values="regret_pct")
    rgt["worst_case"] = rgt.max(1)
    rgt = rgt.sort_values("worst_case")
    rg = pd.concat(reg_rows, ignore_index=True)
    rg["p_holm_in_window"] = rg.groupby(["target", "window"])["p"].transform(holm)
    pooled = pool_regimes(rg)
    mt = rg[(rg.A == "Mamba") & (rg.B == "Transformer")].copy()
    mt["cell"] = mt["rel_pct"].map("{:+.2f}".format) + mt["p_holm_in_window"].map(stars)
    reg_x_win = mt.pivot_table(index=["target", "regime"], columns="window", values="cell", aggfunc="first")

    for name, t in [("A_mse_overall_and_windows", ov), ("B_claims", cl), ("C_dm_by_window", wn),
                    ("D_random_router", rd), ("E_ablation", ab), ("F_regret", rgt.reset_index()),
                    ("G_regimes_by_window", rg), ("H_regimes_pooled", pooled),
                    ("I_regime_x_window_MambaVsTransformer", reg_x_win.reset_index()),
                    ("J_valselect_choice", pd.DataFrame(T_choice))]:
        t.to_csv(OUT / f"{name}.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    print("\n=== MSE: full out-of-sample period and by window ===")
    print(ov.round(5).to_string(index=False))
    print("\n=== Hybrid vs alternatives (full OOS, DM-HAC, Holm within target; p_NI < 0.05 => not worse than margin) ===")
    print(cl.round(4).to_string(index=False))
    print("\n=== Learned vs random router (same Transformer share), anti-router, soft inference ===")
    print(rd.round(5).to_string(index=False))
    if len(ab):
        print(f"\n=== Ablation (seeds {ASEEDS}; delta relative to the full model at the same seeds) ===")
        print(ab.round(5).to_string(index=False))
    print("\n=== Regret relative to the better expert [%] ===")
    print(rgt.round(3).to_string())
    print(f"\nCSV files in {OUT}/ | elapsed {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
