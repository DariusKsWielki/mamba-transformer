"""Training, prediction and the non-neural baselines (Mean, Ridge, HAR)."""
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import gather
from .models import Net


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def parse_model(name):
    """'Hybrid_l0.003' -> ('Hybrid', 0.003); other names use the default penalty."""
    if name.startswith("Hybrid_l"):
        return "Hybrid", float(name[len("Hybrid_l"):])
    return name, C.LAMBDA


@torch.no_grad()
def predict(net, Fg, idx):
    """Hard-routed forecasts and Transformer-use indicators for rows ``idx``."""
    net.eval()
    P, U = [], []
    for k in range(0, len(idx), 1024):
        p, u = net.forward_hard(gather(Fg, idx[k:k + 1024]))
        P.append(p.cpu().numpy())
        U.append(u.cpu().numpy())
    return np.concatenate(P), np.concatenate(U)


def _analysis_outputs(net, Fg, te):
    pm, pt, pr = [], [], []
    net.eval()
    for k in range(0, len(te), 1024):
        a, b, c = net.forward_analysis(gather(Fg, te[k:k + 1024]))
        pm.append(a.cpu().numpy()); pt.append(b.cpu().numpy()); pr.append(c.cpu().numpy())
    return map(np.concatenate, (pm, pt, pr))


def train_eval(model_name, Fg, y_std, tr, va, te, seed):
    """Train Mamba / Transformer / Hybrid(_l<lambda>) and return test and validation forecasts.

    AdamW, MSE loss, gradient clipping, early stopping on the validation MSE. For the hybrid the
    first epoch is an expert warm-up (equal weights) and the usage penalty lambda * mean(g_T)
    is added from epoch WARMUP_EPOCHS on.
    """
    mode, lam = parse_model(model_name)
    set_seed(seed)
    net = Net(Fg.shape[1], mode).to(C.DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=C.LR, weight_decay=C.WD)
    yg = torch.as_tensor(y_std, device=C.DEVICE, dtype=torch.float32)
    best, best_state, bad = 1e9, None, 0
    tr_s = tr[::C.TRAIN_STRIDE]
    for ep in range(C.EPOCHS):
        tau = C.TAU_START + (C.TAU_END - C.TAU_START) * ep / max(1, C.EPOCHS - 1)
        net.train()
        perm = np.random.permutation(tr_s)
        for k in range(0, len(perm) - C.BATCH + 1, C.BATCH):
            b = perm[k:k + C.BATCH]
            pred, use = net.forward_train(gather(Fg, b), tau, expert_warmup=(ep < C.EXPERT_WARMUP_EPOCHS))
            loss = F.mse_loss(pred, yg[b])
            if mode == "Hybrid" and ep >= C.WARMUP_EPOCHS and lam > 0:
                loss = loss + lam * use.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP); opt.step()
        vp, _ = predict(net, Fg, va)
        vl = float(np.mean((vp - y_std[va]) ** 2))
        if mode == "Hybrid" and (ep < C.EXPERT_WARMUP_EPOCHS or (lam > 0 and ep < C.WARMUP_EPOCHS)):
            continue                                   # never keep a state from before routing / the penalty
        if vl < best - 1e-6 or best_state is None:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break
    net.load_state_dict(best_state)
    out = {}
    out["pred"], out["use"] = predict(net, Fg, te)
    out["pred_va"], _ = predict(net, Fg, va)
    if mode == "Hybrid":
        out["pm_int"], out["pt_int"], out["prob"] = _analysis_outputs(net, Fg, te)
        # sanity check: the hard-routed forecast equals the selected expert's output
        hard_check = np.where(out["prob"] > 0.5, out["pt_int"], out["pm_int"])
        out["hard_check_maxdiff"] = float(np.max(np.abs(hard_check - out["pred"])))
    return out


def fit_eval(make_net, Fg, y_std, tr, va, te, seed, hybrid=False, expert_warmup=1, tau_fixed=None):
    """Same protocol as ``train_eval`` for the baselines and the ablation variants."""
    set_seed(seed)
    net = make_net().to(C.DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=C.LR, weight_decay=C.WD)
    yg = torch.as_tensor(y_std, device=C.DEVICE, dtype=torch.float32)
    best, best_state, bad = 1e9, None, 0
    tr_s = tr[::C.TRAIN_STRIDE]
    for ep in range(C.EPOCHS):
        tau = tau_fixed if tau_fixed else C.TAU_START + (C.TAU_END - C.TAU_START) * ep / max(1, C.EPOCHS - 1)
        net.train()
        perm = np.random.permutation(tr_s)
        for k in range(0, len(perm) - C.BATCH + 1, C.BATCH):
            b = perm[k:k + C.BATCH]
            x = gather(Fg, b)
            if hybrid:
                pred, _ = net.forward_train(x, tau, expert_warmup=(ep < expert_warmup))
            else:
                pred, _ = net.forward_hard(x)
            loss = F.mse_loss(pred, yg[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP); opt.step()
        vp, _ = predict(net, Fg, va)
        vl = float(np.mean((vp - y_std[va]) ** 2))
        if hybrid and ep < expert_warmup and ep < C.EPOCHS - 1:
            continue
        if vl < best - 1e-6 or best_state is None:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break
    net.load_state_dict(best_state)
    out = {}
    out["pred"], out["use"] = predict(net, Fg, te)
    out["pred_va"], _ = predict(net, Fg, va)
    if hybrid:
        out["pm_int"], out["pt_int"], out["prob"] = _analysis_outputs(net, Fg, te)
    return out


def cached(path, fn):
    """Load forecasts from ``path`` if present; otherwise compute them with ``fn`` and save."""
    if path.exists():
        return dict(np.load(path))
    t1 = time.time()
    o = fn()
    np.savez(path, **{k: np.asarray(v) for k, v in o.items()})
    print(f"    trained {path.stem:<40} {time.time() - t1:5.0f}s")
    return o


def numpy_baselines(featv, info, y_std, tr, te, target):
    """Mean (zero forecast of the standardised target), Ridge on the last row of features,
    and HAR (daily/weekly-type averages; log-HAR for the realised-volatility target)."""
    res = {"Mean": np.zeros(len(te))}

    def ridge(Xtr, ytr, Xte, lam=1.0):
        A = np.c_[np.ones(len(Xtr)), Xtr]
        B = np.c_[np.ones(len(Xte)), Xte]
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
        return B @ beta

    res["Ridge"] = ridge(featv[tr], y_std[tr], featv[te])
    if target == "abs":     # HAR in the scale of the target: mean |r| over 1 h, 24 h and 120 h
        har = np.c_[info["ar"], info["absr24_hist"], info["absr120_hist"]]
    else:                   # rv24 in logs: log-HAR
        har = np.c_[np.log(info["ar"] + 1e-6), info["rv24_hist"], info["rv120_hist"]]
    mu, sd = har[tr].mean(0), har[tr].std(0) + 1e-9
    res["HAR"] = ridge((har[tr] - mu) / sd, y_std[tr], (har[te] - mu) / sd, lam=1e-3)
    return res
