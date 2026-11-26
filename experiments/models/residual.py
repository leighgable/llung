"""Swappable residual-stream designs for LatentLM (transformer.py).

A design decides how every sublayer (attention, feed-forward) reads from
and writes to the residual stream. Select one in the config:

    model:
      residual: hyper                      # prenorm | scaled | sandwich | postnorm | hyper
      residual_opts: {n: 4, sinkhorn: true}

Interface (write a new design by subclassing these two):

    Residual              one per trunk; owns the stream's SHAPE
      begin(x [B,T,D]) -> state           embeddings / fed-back latents enter here
      end(state) -> [B,T,D]               before the final LayerNorm
      sublayer(index, d_model) -> Sub    index 0..2L-1 (even: attention, odd: ffn)

    Sub (nn.Module)       one per sublayer; owns that sublayer's residual params
      forward(state, fn, norm) -> state  fn: [B,T,D] -> [B,T,D] (the sublayer),
                                          norm: the block's LayerNorm (ln1 / ln2)

The block's LayerNorms stay in Block (keys blocks.i.ln1 / ln2), so the
GPT-2 warm start and old checkpoints load under every design.
"Warm-start safe" below means: at initialization the trunk computes
EXACTLY the pre-LN function (up to LayerNorm eps), so a pretrained
backbone keeps its behaviour and the design only diverges as it trains.

The stream state is per token (no state across positions), so the KV
cache and the latent loop work unchanged with every design: the fed-back
latent enters through begin() like any input embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Residual(nn.Module):
    """Base: a single [B, T, D] stream."""

    def begin(self, x: torch.Tensor):
        return x

    def end(self, state) -> torch.Tensor:
        return state

    def sublayer(self, index: int, d_model: int) -> nn.Module:
        raise NotImplementedError


# ---------------------------------------------------------------- one stream


class _PreNormSub(nn.Module):
    def forward(self, x, fn, norm):
        return x + fn(norm(x))


class PreNorm(Residual):
    """x + f(LN(x)) -- GPT-2 / the original LatentLM. No parameters."""

    def sublayer(self, index, d_model):
        return _PreNormSub()


class _PostNormSub(nn.Module):
    def forward(self, x, fn, norm):
        return norm(x + fn(x))


class PostNorm(Residual):
    """LN(x + f(x)) -- original Transformer. NOT warm-start safe with a
    pre-LN backbone (the sublayers see unnormalized inputs)."""

    def sublayer(self, index, d_model):
        return _PostNormSub()


class _ScaledSub(nn.Module):
    def __init__(self, d_model: int, init: float):
        super().__init__()
        self.gain = nn.Parameter(torch.full((d_model,), float(init)))

    def forward(self, x, fn, norm):
        return x + self.gain * fn(norm(x))


class Scaled(Residual):
    """x + g * f(LN(x)), g a learned per-channel gain (LayerScale).
    init=1.0: warm-start safe. init=0.0: ReZero-style (every block starts
    as the identity -- for training from scratch, it discards a backbone)."""

    def __init__(self, init: float = 1.0):
        super().__init__()
        self.init = init

    def sublayer(self, index, d_model):
        return _ScaledSub(d_model, self.init)


class _SandwichSub(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, fn, norm):
        return x + self.out_norm(fn(norm(x)))


class Sandwich(Residual):
    """x + LN_out(f(LN(x))) -- "peri-LN" / sandwich norm: every write into
    the stream has a learned, bounded scale. NOT warm-start safe (the
    output norm rescales each sublayer's contribution)."""

    def sublayer(self, index, d_model):
        return _SandwichSub(d_model)


# ---------------------------------------------------------- n streams (HC)


def sinkhorn(logits: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """exp(logits) -> doubly stochastic, in log space. The last step
    normalizes columns, so COLUMN sums are exact (what keeps hyper-
    connections pre-LN-equivalent at init); row sums converge with
    `iters` (measured worst case for logits 4I + N(0,1): 4e-2 at 10
    iterations, 2e-2 at 20, 8e-3 at 50)."""
    log_p = logits
    for _ in range(iters):
        log_p = log_p - log_p.logsumexp(-1, keepdim=True)
        log_p = log_p - log_p.logsumexp(-2, keepdim=True)
    return log_p.exp()


class _HyperSub(nn.Module):
    """One sublayer with n residual streams H [..., n, D]:

        read   h   = sum_i a_i H_i            (which streams feed the sublayer)
        write  H'_j = sum_i A_ij H_i + b_j f(LN(h))

    Initialized so that, while the n streams are identical copies (as
    begin() makes them), this is exactly pre-LN: h is a positive multiple
    of the stream (LN is scale-invariant), every column of A sums to 1,
    and b = 1.
    """

    def __init__(self, index: int, n: int, d_model: int, sinkhorn: bool, dynamic: bool, sinkhorn_iters: int = 20):
        super().__init__()
        self.n, self.sinkhorn, self.dynamic, self.sinkhorn_iters = n, sinkhorn, dynamic, sinkhorn_iters
        k = index % n
        if sinkhorn:  # mHC: a = sigmoid, b = 2 sigmoid, A = Sinkhorn(exp(logits))
            read = torch.full((n,), -2.0)
            read[k] = 2.0  # mostly stream k, breaking the symmetry between streams
            self.read = nn.Parameter(read)
            self.write = nn.Parameter(torch.zeros(n))  # 2 sigmoid(0) = 1
            self.mix = nn.Parameter(4.0 * torch.eye(n))  # near-identity, columns sum to 1
        else:  # HC (static): a = e_k, b = 1, A = I
            self.read = nn.Parameter(torch.eye(n)[k].clone())
            self.write = nn.Parameter(torch.ones(n))
            self.mix = nn.Parameter(torch.eye(n))
        if dynamic:  # DHC: input-dependent offsets, zero-initialized
            self.dyn_norm = nn.LayerNorm(d_model)
            self.dyn = nn.Linear(d_model, n + n + n * n, bias=False)
            nn.init.zeros_(self.dyn.weight)
            self.dyn_scale = nn.Parameter(torch.full((3,), 0.01))

    def weights(self, state):
        read, write, mix = self.read, self.write, self.mix
        if self.dynamic:
            n = self.n
            d = torch.tanh(self.dyn(self.dyn_norm(state))).mean(-2)  # [..., n + n + n*n]
            read = read + self.dyn_scale[0] * d[..., :n]
            write = write + self.dyn_scale[1] * d[..., n : 2 * n]
            mix = mix + self.dyn_scale[2] * d[..., 2 * n :].unflatten(-1, (n, n))
        if self.sinkhorn:
            read, write, mix = torch.sigmoid(read), 2 * torch.sigmoid(write), sinkhorn(mix, self.sinkhorn_iters)
        return read, write, mix

    def forward(self, state, fn, norm):  # state [B, T, n, D]
        read, write, mix = self.weights(state)
        h = torch.einsum("...n,...nd->...d", read.to(state.dtype), state)
        out = fn(norm(h))
        mixed = torch.einsum("...ij,...id->...jd", mix.to(state.dtype), state)
        return mixed + write.to(state.dtype).unsqueeze(-1) * out.unsqueeze(-2)


class Hyper(Residual):
    """Hyper-connections (Zhu et al. 2024): n parallel residual streams
    with learned read / write / mixing weights. sinkhorn=True is the
    manifold-constrained variant (mHC: A doubly stochastic, a and b
    non-negative), which keeps the stream norm bounded at depth.
    dynamic=True adds input-dependent offsets (DHC). sinkhorn_iters trades
    exactness of A's row sums for compute (see sinkhorn()). Warm-start safe.

    begin(): copy the input into all n streams; end(): sum them (the
    final LayerNorm makes the n-fold scale irrelevant).
    """

    def __init__(self, n: int = 4, sinkhorn: bool = False, dynamic: bool = False, sinkhorn_iters: int = 20):
        super().__init__()
        self.n, self.use_sinkhorn, self.dynamic = int(n), bool(sinkhorn), bool(dynamic)
        self.sinkhorn_iters = int(sinkhorn_iters)

    def begin(self, x):
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.n, x.shape[-1])

    def end(self, state):
        return state.sum(-2)

    def sublayer(self, index, d_model):
        return _HyperSub(index, self.n, d_model, self.use_sinkhorn, self.dynamic, self.sinkhorn_iters)


RESIDUALS = {
    "prenorm": PreNorm,
    "postnorm": PostNorm,
    "scaled": Scaled,
    "sandwich": Sandwich,
    "hyper": Hyper,
}


def make_residual(name: str, **opts) -> Residual:
    if name not in RESIDUALS:
        raise ValueError(f"unknown residual '{name}' (one of: {', '.join(RESIDUALS)})")
    return RESIDUALS[name](**opts)
