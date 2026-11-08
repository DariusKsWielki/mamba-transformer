"""Data loading, targets, walk-forward splits and scaling."""
import numpy as np
import pandas as pd
import torch

from . import config as C

PAPER_FEATURES = ['Open', 'High', 'Low', 'Close', 'Vol', 'SMA_24', 'SMA_50', 'EMA_12',
                  'MACD', 'MACD_Signal', 'RSI', 'ROC_12', 'BB_Upper', 'BB_Lower',
                  'BB_Width', 'ATR', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos']
# price-level inputs are expressed relative to the close: log(x_t / C_t)
PRICE_LIKE = ['Open', 'High', 'Low', 'SMA_24', 'SMA_50', 'EMA_12', 'BB_Upper', 'BB_Lower']


def parse_dates_robust(s):
    """Parse timestamps, also repairing malformed entries such as '2003-09-03 03:0'
    (only the date and hour matter for hourly data)."""
    s = s.astype(str).str.strip()
    ex = s.str.extract(r"^(\d{4})-(\d{1,2})-(\d{1,2})[ T]?(\d{1,2})?(?::(\d{1,2}))?")
    ok = ex[0].notna()
    parts = ex.fillna(0).astype(int)
    out = pd.to_datetime(dict(year=parts[0].where(ok, 1970), month=parts[1].where(ok, 1),
                              day=parts[2].where(ok, 1), hour=parts[3], minute=parts[4]),
                         errors="coerce")
    out[~ok] = pd.NaT
    std = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    n_odd, n_bad = int((std.isna() & out.notna()).sum()), int(out.isna().sum())
    if n_odd or n_bad:
        print(f"Timestamps: {n_odd} non-standard entries repaired, {n_bad} unreadable entries dropped.")
    return out


def load_data(csv_file=None):
    """Return (df, features, targets, info).

    Targets are defined for row t = last step of the input window:
      abs : |r_{t+1}|
      rv24: log( sqrt( mean(r_{t+1..t+24}^2) ) + 1e-6 )
    """
    df = pd.read_csv(csv_file or C.CSV_FILE).replace([np.inf, -np.inf], np.nan)
    df[C.DATE_COL] = parse_dates_robust(df[C.DATE_COL])
    n0 = len(df)
    df = df.dropna(subset=[C.DATE_COL]).drop_duplicates(subset=[C.DATE_COL], keep="first")
    df = df.sort_values(C.DATE_COL).dropna().reset_index(drop=True)
    print(f"Rows after cleaning timestamps/NaN/duplicates: {len(df):,} (removed {n0 - len(df):,})")
    cols = [c for c in PAPER_FEATURES if c in df.columns]
    missing = [c for c in PAPER_FEATURES if c not in df.columns]
    if missing:
        print("Missing columns (skipped):", missing)

    close = df["Close"].values.astype(float)
    r = np.r_[0.0, np.diff(np.log(close))]                     # r[t] = log C_t - log C_{t-1}
    ar = np.abs(r)
    rv24_hist = np.log(np.sqrt(pd.Series(r ** 2).rolling(24, min_periods=6).mean().bfill().values) + 1e-6)
    absr24_hist = pd.Series(ar).rolling(24, min_periods=6).mean().bfill().values
    absr120_hist = pd.Series(ar).rolling(120, min_periods=24).mean().bfill().values
    rv120_hist = np.log(np.sqrt(pd.Series(r ** 2).rolling(120, min_periods=24).mean().bfill().values) + 1e-6)
    atr_raw = df["ATR"].values.astype(float)

    feat = pd.DataFrame(index=df.index)
    for c in cols:                                              # stationary transformations
        v = df[c].values.astype(float)
        if c in PRICE_LIKE:
            v = np.log(np.maximum(v, 1e-9) / close)
        elif c == "Close":
            continue
        elif c in ("ATR", "MACD", "MACD_Signal"):
            v = v / close
        elif c == "Vol":
            v = np.log1p(np.maximum(v, 0))
        feat[c] = v
    feat["r_hist"] = r
    feat["absr_hist"] = ar
    feat["rv24_hist"] = rv24_hist
    feat["rv120_hist"] = rv120_hist

    T = len(df)
    tg = {"abs": np.r_[ar[1:], np.nan]}
    fut = np.full(T, np.nan)
    cs = np.r_[0, np.cumsum(r ** 2)]
    for_t = np.arange(T - 24)
    fut[for_t] = (cs[for_t + 25] - cs[for_t + 1]) / 24           # mean r^2 over t+1..t+24
    tg["rv24"] = np.log(np.sqrt(fut) + 1e-6)

    hours = df[C.DATE_COL].dt.hour.values
    info = dict(dates=df[C.DATE_COL].values, close=close, r=r, ar=ar, atr=atr_raw,
                rv24_hist=rv24_hist, rv120_hist=rv120_hist,
                absr24_hist=absr24_hist, absr120_hist=absr120_hist,
                hour_sin=np.sin(2 * np.pi * hours / 24), hour_cos=np.cos(2 * np.pi * hours / 24))
    if "log_ret" in df:
        info["log_ret_col"] = df["log_ret"].values.astype(float)
    return df, feat, tg, info


def alignment_check(info):
    """Leakage check: the target must be the return of the NEXT hour, not one already in the window."""
    r, rows = info["r"], {}
    if "log_ret_col" in info:
        lr = info["log_ret_col"]
        rows["corr(log_ret_col[t], r_t)"] = np.corrcoef(lr[1:], r[1:])[0, 1]
        rows["corr(log_ret_col[t], r_{t+1})"] = np.corrcoef(lr[1:-1], r[2:])[0, 1]
        rows["corr(log_ret_col[t], r_{t-1})"] = np.corrcoef(lr[2:], r[1:-1])[0, 1]
    rows["autocorr r (lag1)"] = np.corrcoef(r[2:], r[1:-1])[0, 1]
    s = pd.Series(rows)
    print("\n=== Target alignment check ===")
    print(s.round(4).to_string())
    return s


def window_indices(dates, tgt, ts, te, H):
    """Training and test rows for one walk-forward window.

    A training row is kept only if its whole target horizon ends before the test start,
    and only if it lies within the preceding TRAIN_YEARS years.
    """
    d = pd.to_datetime(dates)
    ts, te = pd.Timestamp(ts), pd.Timestamp(te)
    T = len(d)
    t = np.arange(C.SEQ_LEN - 1, T - H)
    t = t[~np.isnan(tgt[t])]
    first_target_time = d[np.minimum(t + 1, T - 1)]
    last_target_time = d[np.minimum(t + H, T - 1)]
    tr = t[last_target_time < ts]
    if C.TRAIN_YEARS:
        tr = tr[d[tr] >= ts - pd.DateOffset(years=C.TRAIN_YEARS)]
    te_idx = t[(first_target_time >= ts) & (last_target_time < te)]
    return tr, te_idx


def robust_scale(featv, train_rows):
    """Median/IQR scaling fitted on the training period only, clipped to +/- CLIP_Z."""
    med = np.median(featv[train_rows], 0)
    iqr = np.subtract(*np.percentile(featv[train_rows], [75, 25], 0))
    iqr[iqr < 1e-9] = 1.0
    return np.clip((featv - med) / iqr, -C.CLIP_Z, C.CLIP_Z).astype(np.float32)


def gather(Fg, idx):
    """Input windows of length SEQ_LEN ending at rows ``idx``."""
    ar = torch.arange(-C.SEQ_LEN + 1, 1, device=Fg.device)
    return Fg[torch.as_tensor(idx, device=Fg.device)[:, None] + ar]


def prepare_window(featv_raw, TG, info, target, w):
    """All arrays needed to train and evaluate models in walk-forward window ``w``.

    Returns None when the window has too little data.
    """
    H = C.TARGET_H[target]
    ts, te_ = C.WINDOWS[w]
    tr, te = window_indices(info["dates"], TG[target], ts, te_, H)
    if len(te) < 500 or len(tr) < 5000:
        print(f"[{target} w{w}] not enough data (train {len(tr)}, test {len(te)}), skipped")
        return None
    n_va = int(len(tr) * C.VAL_FRAC)
    tr_fit, va = tr[:-n_va - H], tr[-n_va:]          # gap of H rows: no target overlap with validation
    train_rows = np.arange(0, tr_fit[-1] + 1)
    featv = robust_scale(featv_raw, train_rows)
    y = TG[target].copy()
    mu, sd = np.nanmean(y[tr_fit]), np.nanstd(y[tr_fit])
    y_std = np.nan_to_num(np.clip((y - mu) / sd, -C.CLIP_Z, C.CLIP_Z))
    return dict(target=target, w=w, H=H, ts=ts, te_end=te_, tr_fit=tr_fit, va=va, te=te,
                train_rows=train_rows, featv=featv, y_std=y_std, mu=mu, sd=sd,
                Fg=torch.as_tensor(featv, device=C.DEVICE))
