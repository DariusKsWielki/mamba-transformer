"""Models: Mamba and Transformer experts, the routed hybrid, and the deep-learning baselines.

The selective state-space (Mamba/S6) block is implemented in plain PyTorch (no ``mamba_ssm``
CUDA kernels), so the code runs on any device.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class SelectiveSSM(nn.Module):
    """Selective state-space block (Mamba, S6). Returns only the last time step,
    which is all a one-step forecast needs."""

    def __init__(self, dm, ds=None, dc=None, ex=None):
        super().__init__()
        ds = ds or C.D_STATE
        dc = dc or C.D_CONV
        ex = ex or C.EXPAND
        self.di, self.ds, self.dtr = ex * dm, ds, max(1, math.ceil(dm / 16))
        self.inp = nn.Linear(dm, 2 * self.di, bias=False)
        self.cv = nn.Conv1d(self.di, self.di, dc, groups=self.di, padding=dc - 1)
        self.xp = nn.Linear(self.di, self.dtr + 2 * ds, bias=False)
        self.dtp = nn.Linear(self.dtr, self.di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, ds + 1, dtype=torch.float32)).repeat(self.di, 1))
        self.D = nn.Parameter(torch.ones(self.di))
        self.op = nn.Linear(self.di, dm, bias=False)
        # dt initialisation as in the original Mamba (dt in [1e-3, 1e-1])
        with torch.no_grad():
            nn.init.uniform_(self.dtp.weight, -self.dtr ** -0.5, self.dtr ** -0.5)
            dt = torch.exp(torch.rand(self.di) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dtp.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        B, L, _ = x.shape
        xr, z = self.inp(x).chunk(2, -1)
        xr = F.silu(self.cv(xr.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.xp(xr), [self.dtr, self.ds, self.ds], -1)
        dt = F.softplus(self.dtp(dt))                               # B, L, di
        A = -torch.exp(self.A_log)                                  # di, ds
        h = x.new_zeros(B, self.di, self.ds)
        for t in range(L):                                          # sequential selective scan
            dA = torch.exp(dt[:, t, :, None] * A)
            h = dA * h + dt[:, t, :, None] * Bm[:, t, None, :] * xr[:, t, :, None]
        y = (h * Cm[:, -1, None, :]).sum(-1) + xr[:, -1] * self.D
        return self.op(y * F.silu(z[:, -1]))                        # B, dm


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block; returns the last time step."""

    def __init__(self, d, h=None, dr=None):
        super().__init__()
        h = h or C.HEADS
        dr = C.DROPOUT if dr is None else dr
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.at = nn.MultiheadAttention(d, h, dropout=dr, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dr))

    def forward(self, x):
        a = self.n1(x)
        x = x + self.at(a, a, a, need_weights=False)[0]
        x = x + self.ff(self.n2(x))
        return x[:, -1]                                             # B, d


class Net(nn.Module):
    """Shared embedding + expert(s) + shared linear head.

    mode = "Mamba" | "Transformer" | "Hybrid". In the hybrid, a router MLP reads the last
    embedded step and selects one expert per input.
    """

    def __init__(self, fin, mode):
        super().__init__()
        self.mode = mode
        self.ip, self.ln = nn.Linear(fin, C.HID), nn.LayerNorm(C.HID)
        self.pe = nn.Parameter(torch.randn(1, C.SEQ_LEN, C.HID) * 0.01)
        self.head = nn.Linear(C.HID, 1)
        if mode in ("Mamba", "Hybrid"):
            self.mn, self.mb = nn.LayerNorm(C.HID), SelectiveSSM(C.HID)
        if mode in ("Transformer", "Hybrid"):
            self.tb = TransformerBlock(C.HID)
        if mode == "Hybrid":
            self.router = nn.Sequential(nn.Linear(C.HID, C.ROUTER_HID), nn.ReLU(), nn.Linear(C.ROUTER_HID, 2))

    def embed(self, x):
        return self.ln(self.ip(x)) + self.pe

    def mamba_last(self, h):
        return h[:, -1] + self.mb(self.mn(h))                       # residual connection

    def trans_last(self, h):
        return self.tb(h)

    def router_prob(self, h):
        """Router probability of choosing the Transformer."""
        return F.softmax(self.router(h[:, -1]), -1)[:, 1]

    def forward_hard(self, x):
        """Inference with HARD routing: only the selected expert is executed for each input.
        Returns (forecast, indicator of Transformer use)."""
        h = self.embed(x)
        if self.mode == "Mamba":
            return self.head(self.mamba_last(h)).squeeze(-1), torch.zeros(len(x), device=x.device)
        if self.mode == "Transformer":
            return self.head(self.trans_last(h)).squeeze(-1), torch.ones(len(x), device=x.device)
        sel = self.router_prob(h) > 0.5
        out = h.new_zeros(len(x), C.HID)
        if (~sel).any():
            out[~sel] = self.mamba_last(h[~sel])
        if sel.any():
            out[sel] = self.trans_last(h[sel])
        return self.head(out).squeeze(-1), sel.float()

    def forward_train(self, x, tau=1.0, expert_warmup=False, hard=None):
        """Training pass. For the hybrid: straight-through Gumbel-Softmax gate
        (or a fixed 0.5/0.5 mix during the expert warm-up).
        Returns (forecast, soft probability of the Transformer)."""
        if self.mode != "Hybrid":
            return self.forward_hard(x)
        hard = (C.ROUTER_TRAIN == "st") if hard is None else hard
        h = self.embed(x)
        logits = self.router(h[:, -1])
        w = torch.full_like(logits, 0.5) if expert_warmup else F.gumbel_softmax(logits, tau=tau, hard=hard)
        out = w[:, :1] * self.mamba_last(h) + w[:, 1:] * self.trans_last(h)
        return self.head(out).squeeze(-1), F.softmax(logits, -1)[:, 1]

    @torch.no_grad()
    def forward_analysis(self, x):
        """Analysis only (both experts through the shared head); NOT the inference mode."""
        h = self.embed(x)
        return (self.head(self.mamba_last(h)).squeeze(-1), self.head(self.trans_last(h)).squeeze(-1),
                self.router_prob(h))


class HybridVariant(Net):
    """Hybrid with one component changed, used in the ablation study. Inference is always hard."""

    def __init__(self, fin, cfg):
        super().__init__(fin, "Hybrid")
        self.cfg = cfg
        if cfg.get("router") == "linear":
            self.router = nn.Linear(C.HID, 2)

    def forward_train(self, x, tau=1.0, expert_warmup=False, hard=None):
        return super().forward_train(x, tau, expert_warmup, hard=(self.cfg.get("train", "st") == "st"))


class DLinearNet(nn.Module):
    """DLinear / NLinear (Zeng et al., 2023): a linear map over time for each input channel
    (weights shared across channels), followed by a linear combination of the channels."""

    def __init__(self, fin, kind="dlinear", kernel=25):
        super().__init__()
        self.kind, self.k = kind, kernel
        self.lin_s = nn.Linear(C.SEQ_LEN, 1)
        self.lin_t = nn.Linear(C.SEQ_LEN, 1)
        self.mix = nn.Linear(fin, 1)

    def forward_hard(self, x):
        z = x.transpose(1, 2)                                       # B, C, L
        if self.kind == "nlinear":
            last = z[:, :, -1:]
            o = self.lin_s(z - last).squeeze(-1) + last.squeeze(-1)
        else:
            p = (self.k - 1) // 2
            trend = F.avg_pool1d(F.pad(z, (p, self.k - 1 - p), mode="replicate"), self.k, stride=1)
            o = self.lin_s(z - trend).squeeze(-1) + self.lin_t(trend).squeeze(-1)
        return self.mix(o).squeeze(-1), torch.zeros(len(x), device=x.device)


class PatchTSTNet(nn.Module):
    """PatchTST (Nie et al., 2023), channel-independent: patches of length 16 with stride 8,
    one encoder layer (as in the Transformer expert), then a linear combination of the channels."""

    def __init__(self, fin, P=16, S=8):
        super().__init__()
        self.P, self.S = P, S
        self.n = (C.SEQ_LEN - P) // S + 1
        self.proj = nn.Linear(P, C.HID)
        self.pe = nn.Parameter(torch.randn(1, self.n, C.HID) * 0.02)
        self.enc = nn.TransformerEncoderLayer(C.HID, C.HEADS, 4 * C.HID, C.DROPOUT,
                                              batch_first=True, norm_first=True)
        self.flat = nn.Linear(self.n * C.HID, 1)
        self.mix = nn.Linear(fin, 1)

    def forward_hard(self, x):
        B, L, Ch = x.shape
        z = x.permute(0, 2, 1).reshape(B * Ch, L).unfold(1, self.P, self.S)   # B*C, n, P
        z = self.enc(self.proj(z) + self.pe)
        z = self.flat(z.reshape(B * Ch, -1)).reshape(B, Ch)
        return self.mix(z).squeeze(-1), torch.zeros(B, device=x.device)
