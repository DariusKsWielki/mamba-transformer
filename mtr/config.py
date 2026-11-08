"""Experiment configuration.

All values reproduce the settings reported in the paper (Section 5 and Table 10).
Scripts may override individual attributes at run time (e.g. ``config.SEEDS``),
so the rest of the package always reads them through this module.
"""
from pathlib import Path

import torch

# --------------------------------------------------------------------------- data
CSV_FILE = "data/gbpusd_1h_features.csv"
DATE_COL = "DateTime"
OUT = Path("results")

TARGETS = ["abs", "rv24"]                       # y_abs (|r_{t+1}|) and y_rv24 (log RV over t+1..t+24)
TARGET_H = {"abs": 1, "rv24": 24}               # forecast horizon in hours (used by HAC and the bootstrap)

# Walk-forward evaluation windows: (start of test, end of test). Training uses data before the test start.
WINDOWS = [("2021-04-01", "2022-04-01"),
           ("2022-04-01", "2023-04-01"),
           ("2023-04-01", "2025-01-01")]

# Seed sets: exploratory runs fixed the hypotheses, confirmatory runs test them.
EXPLORATORY_SEEDS = [0, 1, 2]
CONFIRMATORY_SEEDS = [3, 4, 5]
SEEDS = CONFIRMATORY_SEEDS

# Models: "Hybrid_l<lambda>" trains the hybrid with an attention-usage penalty lambda (Section 4).
MODELS = ["Mamba", "Transformer", "Hybrid"]
LAMBDA_GRID = [0.001, 0.003, 0.01]

# --------------------------------------------------------------------------- architecture
SEQ_LEN = 48            # input window (hours)
HID = 64                # hidden dimension
HEADS = 4               # attention heads
ROUTER_HID = 32         # router hidden size
D_STATE, D_CONV, EXPAND = 8, 4, 2   # selective state-space (Mamba) block
DROPOUT = 0.1

# --------------------------------------------------------------------------- training
EPOCHS, PATIENCE, BATCH, LR, WD = 6, 2, 256, 3e-4, 1e-4
GRAD_CLIP = 1.0
TRAIN_YEARS = 8         # training data: the 8 years preceding each test window
TRAIN_STRIDE = 2        # every 2nd training window is used
VAL_FRAC = 0.1          # final 10% of the training period (chronological) for early stopping
CLIP_Z = 5.0            # scaled features and standardised target are clipped to +/- 5

# --------------------------------------------------------------------------- router
ROUTER_TRAIN = "st"     # straight-through Gumbel-Softmax (hard forward pass, soft gradients)
TAU_START, TAU_END = 1.0, 0.5       # Gumbel temperature, annealed linearly over the epochs
LAMBDA = 0.0            # default attention-usage penalty
WARMUP_EPOCHS = 1       # the penalty is applied from this epoch on
EXPERT_WARMUP_EPOCHS = 1  # first epoch: both experts trained with equal weights (0.5/0.5)

# --------------------------------------------------------------------------- statistics
NI_MARGIN = 0.02        # non-inferiority margin (share of the comparator's MSE)
NI_MARGINS = [0.005, 0.02]
B_BOOT = 1000           # block-bootstrap replications for the regime analysis
N_RANDOM_DRAWS = 200    # draws of the random router (permutation test)
EVENT_WINDOWS = [("2022-09-23", "2022-10-15", "mini_budget_2022")]

# --------------------------------------------------------------------------- extended analyses
BASELINE_SEEDS = [3, 4, 5]
ABL_SEEDS = [3]
ROUTER_VARIANTS = {
    "Abl_LinearRouter":   dict(router="linear"),
    "Abl_SoftGumbel":     dict(train="soft"),
    "Abl_NoExpertWarmup": dict(expert_warmup=0),
    "Abl_Tau1.0":         dict(tau_fixed=1.0),
    "Abl_Tau0.2":         dict(tau_fixed=0.2),
}
FORECAST_PLOT = ("2022-09-15", "2022-10-21")   # gilt-crisis period shown in Fig 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
