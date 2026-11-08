"""Aggregate HistData.com one-minute GBP/USD bars to hourly OHLC bars.

Download the free "Generic ASCII" M1 files for GBP/USD (e.g. DAT_ASCII_GBPUSD_M1_2000.csv ...
DAT_ASCII_GBPUSD_M1_2024.csv) from https://www.histdata.com, put them in one folder and run

    python aggregate_histdata.py --input-dir raw_histdata --output data/gbpusd_1h_ohlc.csv
    python build_features.py --input data/gbpusd_1h_ohlc.csv --output data/gbpusd_1h_features.csv

HistData timestamps are in Eastern Standard Time without daylight-saving adjustment. Each hourly bar
is labelled with the start of the hour; hours without any one-minute bar are omitted.
"""
import argparse
from pathlib import Path

import pandas as pd


def read_m1(path):
    df = pd.read_csv(path, sep=";", header=None, names=["dt", "Open", "High", "Low", "Close", "Vol"])
    df["DateTime"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    return df.drop(columns="dt")


def to_hourly(m1):
    m1 = m1.sort_values("DateTime").drop_duplicates("DateTime").set_index("DateTime")
    h = m1.resample("1h", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Vol": "sum"})
    return h.dropna(subset=["Close"]).reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="DAT_ASCII_GBPUSD_M1_*.csv")
    ap.add_argument("--output", default="data/gbpusd_1h_ohlc.csv")
    ap.add_argument("--end", default="2024-12-31 23:59:59", help="last timestamp kept")
    a = ap.parse_args()
    files = sorted(Path(a.input_dir).glob(a.pattern))
    if not files:
        raise SystemExit(f"No files matching {a.pattern} in {a.input_dir}")
    h = to_hourly(pd.concat([read_m1(f) for f in files], ignore_index=True))
    h = h[h["DateTime"] <= pd.Timestamp(a.end)]
    h.to_csv(a.output, index=False)
    print(f"{len(h):,} hourly bars written to {a.output} ({h.DateTime.iloc[0]} to {h.DateTime.iloc[-1]})")
