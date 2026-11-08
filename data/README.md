# Data

The experiments read `data/gbpusd_1h_features.csv`: hourly GBP/USD bars from 1 June 2000 to
31 December 2024 (149,523 rows), with timestamps in Eastern Standard Time without daylight-saving adjustment
(the trading week opens on Sunday at 17:00 and closes on Friday at 17:00).

Columns (the first, unnamed column is a row index):

| Column | Description |
|---|---|
| `DateTime` | timestamp of the hourly bar, `YYYY-MM-DD HH:MM:SS` |
| `Open`, `High`, `Low`, `Close` | hourly open, high, low and close prices |
| `Vol` | tick volume (zero in this data set; the feature is kept for completeness) |
| `log_ret` | log(C_t / C_{t-1}) |
| `SMA_24`, `SMA_50`, `EMA_12` | moving averages of the close |
| `MACD`, `MACD_Signal` | EMA_12 − EMA_26 and its 9-period EMA |
| `RSI` | 14-period RSI (Wilder) |
| `ROC_12` | 12-period rate of change [%] |
| `BB_Upper`, `BB_Lower`, `BB_Width` | 20-period Bollinger bands (2 population SD) and width [%] |
| `ATR` | 14-period average true range (Wilder) |
| `Hour_Sin`, `Hour_Cos`, `Day_Sin`, `Day_Cos` | calendar encodings |

`build_features.py` recreates every indicator column from the OHLC columns (agreement with the file
used in the paper is within 1e-5 after the indicator warm-up).

## Source

The prices are the free one-minute GBP/USD bars published by HistData.com
(https://www.histdata.com, "Generic ASCII" M1 files), aggregated to hourly bars. HistData timestamps
are in Eastern Standard Time without daylight-saving adjustment. The raw files are not redistributed
here; download them from HistData.com and rebuild the input file with

```bash
python aggregate_histdata.py --input-dir raw_histdata --output data/gbpusd_1h_ohlc.csv
python build_features.py --input data/gbpusd_1h_ohlc.csv --output data/gbpusd_1h_features.csv
```
