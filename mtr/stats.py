"""Forecast-comparison statistics: Diebold-Mariano with HAC variance, non-inferiority,
Holm / Benjamini-Hochberg corrections, block-bootstrap regime tests and the FLOP model."""
import math

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def nw_lrv(x, lags):
    """Newey-West long-run variance with Bartlett weights."""
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    s = x @ x / T
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / T
    return max(s, 1e-300)


def default_lags(T, h):
    return max(h - 1, int(4 * (T / 100) ** (2 / 9)))


def dm_test(la, lb, h):
    """Diebold-Mariano test of equal squared-error loss, HAC variance and the
    Harvey-Leybourne-Newbold small-sample correction. Returns (mean diff, statistic, two-sided p)."""
    d = la - lb
    T = len(d)
    se = np.sqrt(nw_lrv(d, default_lags(T, h)) / T)
    stat = d.mean() / se * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return d.mean(), stat, 2 * stats.t.sf(abs(stat), T - 1)


def ni_pvalue(la, lb, h, margin):
    """Non-inferiority: H0 says A is worse than B by at least margin * MSE(B).
    p < 0.05 means A is not worse than B by more than the margin."""
    d = la - lb
    T = len(d)
    se = np.sqrt(nw_lrv(d, default_lags(T, h)) / T)
    return stats.t.cdf((d.mean() - margin * lb.mean()) / se, T - 1)


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    adj, run = np.empty(m), 0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1, run)
    return adj


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    adj, run = np.empty(m), 1
    for k, i in enumerate(np.argsort(p)[::-1]):
        run = min(run, p[i] * m / (m - k))
        adj[i] = run
    return adj


def boot_idx(T, block, B, seed=0):
    """Moving-block bootstrap indices."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    st = rng.integers(0, T - block + 1, size=(B, nb))
    return (st[:, :, None] + np.arange(block)).reshape(B, -1)[:, :T].astype(np.int32)


def make_regimes(info, te, train_rows, prob=None):
    """Pre-defined regimes of the forecast hour: volatility terciles (training thresholds),
    top ATR decile, trading sessions, Mondays/Fridays and the event windows."""
    d = pd.to_datetime(info["dates"])
    tt = d[te + 1]
    hr, wd = tt.hour.values, tt.weekday.values
    rv = info["rv24_hist"][te]
    q1, q2 = np.quantile(info["rv24_hist"][train_rows], [1 / 3, 2 / 3])
    a90 = np.quantile(info["atr"][train_rows], 0.9)
    R = {"vol_low": rv <= q1, "vol_mid": (rv > q1) & (rv <= q2), "vol_high": rv > q2,
         "atr_top10": info["atr"][te] > a90,
         "sess_asia_00-07": hr < 7, "sess_london_07-12": (hr >= 7) & (hr < 12),
         "sess_overlap_12-16": (hr >= 12) & (hr < 16), "sess_ny_16-21": (hr >= 16) & (hr < 21),
         "sess_late_21-24": hr >= 21, "monday": wd == 0, "friday": wd == 4}
    for a, b, name in C.EVENT_WINDOWS:
        R[name] = (tt >= pd.Timestamp(a)) & (tt < pd.Timestamp(b))
    if prob is not None:
        R["router_T"] = prob > 0.5
        R["router_M"] = prob <= 0.5
    return {k: np.asarray(v, bool) for k, v in R.items()}


def regime_tests(L, regimes, pairs, H, min_n=150):
    """Mean loss difference within each regime, block-bootstrap CI and p-value."""
    T = len(next(iter(L.values())))
    I = boot_idx(T, max(24, 2 * H), C.B_BOOT)
    rows = []
    for rn, m in regimes.items():
        if m.sum() < min_n:
            continue
        mI = m[I]
        den = mI.sum(1)
        ok = den > 0
        for a, b in pairs:
            d = L[a] - L[b]
            obs = d[m].mean()
            bt = ((d * m)[I].sum(1)[ok]) / den[ok]
            c = bt - bt.mean()
            p = (np.sum(np.abs(c) >= abs(obs)) + 1) / (len(c) + 1)
            rows.append(dict(regime=rn, n=int(m.sum()), A=a, B=b, diff=obs,
                             rel_pct=100 * obs / L[b][m].mean(),
                             ci_lo=np.quantile(bt, .025), ci_hi=np.quantile(bt, .975), p=p))
    return pd.DataFrame(rows)


def pool_regimes(rg):
    """Combine regime tests across windows (Stouffer z with sign) and apply Holm / BH.
    A result is 'robust' if Holm-adjusted p < 0.05 and the sign agrees in every window."""
    g = []
    for (tg_, r_, a, b), x in rg.groupby(["target", "regime", "A", "B"]):
        z = np.sign(x["diff"]) * stats.norm.isf(np.clip(x["p"], 1e-12, 1) / 2)
        zs = z.sum() / np.sqrt(len(z))
        g.append(dict(target=tg_, regime=r_, A=a, B=b, windows=len(x),
                      same_sign=int(max((x["diff"] > 0).sum(), (x["diff"] < 0).sum())),
                      mean_rel_pct=x["rel_pct"].mean(), z=zs, p=2 * stats.norm.sf(abs(zs))))
    pooled = pd.DataFrame(g)
    pooled["p_holm"] = pooled.groupby("target")["p"].transform(holm)
    pooled["p_BH"] = pooled.groupby("target")["p"].transform(bh)
    pooled["robust"] = (pooled["p_holm"] < 0.05) & (pooled["same_sign"] == pooled["windows"])
    return pooled.sort_values(["target", "A", "B", "p"])


def flops_table():
    """Theoretical forward cost per input (FLOPs) of the experts, the router and the
    hard-routed hybrid at several Transformer-usage shares (Table A)."""
    rows = []
    for N in [48, 96, 192, 336, 720]:
        d, di = C.HID, C.EXPAND * C.HID
        tr = 2 * 4 * N * d * d + 4 * N * N * d + 2 * 2 * N * d * 4 * d
        mb = 2 * N * d * 2 * di + 2 * di * C.D_CONV * N + 2 * N * di * (max(1, math.ceil(d / 16)) + 2 * C.D_STATE) \
            + 6 * N * di * C.D_STATE + 2 * di * d
        rt = 2 * (d * C.ROUTER_HID + C.ROUTER_HID * 2)
        row = dict(N=N, Transformer_MFLOP=tr / 1e6, Mamba_MFLOP=mb / 1e6,
                   attention_share_pct=100 * 4 * N * N * d / tr)
        for p in [0.05, 0.15, 0.30]:
            row[f"Hybrid_p{p:.2f}_MFLOP"] = (rt + (1 - p) * mb + p * tr) / 1e6
        rows.append(row)
    return pd.DataFrame(rows)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
