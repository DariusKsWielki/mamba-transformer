"""Build the model input file from hourly OHLC bars.

Input : CSV with columns DateTime, Open, High, Low, Close and optionally Vol (hourly bars).
Output: the feature file read by the experiments (data/gbpusd_1h_features.csv).

Indicators (all computed on the close unless stated otherwise):
  log_ret        log(C_t / C_{t-1})
  SMA_24, SMA_50 simple moving averages
  EMA_12         exponential moving average (span 12)
  MACD           EMA_12 - EMA_26;  MACD_Signal = EMA_9 of MACD
  RSI            14-period RSI with Wilder smoothing
  ROC_12         12-period rate of change [%]
  BB_Upper/Lower 20-period Bollinger bands, 2 population standard deviations
  BB_Width       (BB_Upper - BB_Lower) / middle band * 100
  ATR            14-period average true range with Wilder smoothing (uses High and Low)
  Hour_*, Day_*  sine/cosine encodings of the hour of day and the day of week
Rows with incomplete indicator windows are dropped.

    python build_features.py --input data/gbpusd_1h_ohlc.csv --output data/gbpusd_1h_features.csv
"""
import argparse

import numpy as np
import pandas as pd


def build_features(df):
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").drop_duplicates("DateTime").reset_index(drop=True)
    if "Vol" not in df:
        df["Vol"] = 0
    c = df["Close"]
    df["log_ret"] = np.log(c / c.shift())
    df["SMA_24"] = c.rolling(24).mean()
    df["SMA_50"] = c.rolling(50).mean()
    df["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - c.ewm(span=26, adjust=False).mean()
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    df["ROC_12"] = c.pct_change(12) * 100
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    df["BB_Upper"], df["BB_Lower"] = mid + 2 * sd, mid - 2 * sd
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / mid * 100
    prev = c.shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    hour, dow = df["DateTime"].dt.hour, df["DateTime"].dt.dayofweek
    df["Hour_Sin"], df["Hour_Cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    df["Day_Sin"], df["Day_Cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    cols = ["DateTime", "Open", "High", "Low", "Close", "Vol", "log_ret", "SMA_24", "SMA_50", "EMA_12",
            "MACD", "MACD_Signal", "RSI", "ROC_12", "BB_Upper", "BB_Lower", "BB_Width", "ATR",
            "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos"]
    return df[cols].dropna().reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/gbpusd_1h_features.csv")
    a = ap.parse_args()
    out = build_features(pd.read_csv(a.input))
    out.to_csv(a.output)
    print(f"{len(out):,} rows written to {a.output}")
