"""Swappable latent feedback for LatentLM: what enters the BASE of the
residual stream at latent step t+1, given the trunk's output at step t.

    model:
      feedback: expected                   # projected | expected
      feedback_opts: {tau: 1.0, top_k: null}

Interface: an nn.Module with forward(h, model) -> x, where h is the
final-LayerNorm hidden state ([B, D] or [B, T, D]) and x has the same
shape and lives where token embeddings live. `model` gives access to
the tied embedding / LM head without registering them twice.

Scale matters: x enters the stream unnormalized and is carried through
every layer, so it must be on the scale of a token embedding (rms ~0.036
for the codeparrot-small warm start). Measured on the warm-started
model, a TRUE token embedding rescaled to unit std raises the next-token
loss from 3.05 to 9.18 -- worse than feeding zeros (6.46).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projected(nn.Module):
    """x = W h, then rescaled. A learned map from the output space back to
    the input space (Coconut-style), randomly initialized -- the warm
    start does not touch it.

    scale="unit_std"  : x / std(x)  -- the ORIGINAL LatentLM behaviour;
                        rms 1.0, ~28x a token embedding (see module doc)
    scale="embedding" : x scaled to the token embeddings' rms
    scale="none"      : raw W h
    """

    def __init__(self, d_model: int, scale: str = "unit_std"):
        super().__init__()
        if scale not in ("unit_std", "embedding", "none"):
            raise ValueError(f"unknown scale '{scale}' (unit_std | embedding | none)")
        self.scale = scale
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h, model):
        x = self.proj(h)
        if self.scale == "unit_std":
            x = x / x.std(dim=-1, keepdim=True).clamp(min=1e-5)
        elif self.scale == "embedding":
            target = model.tok_emb.weight.detach().pow(2).mean().sqrt()
            x = x * (target / x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8))
        return x


class Expected(nn.Module):
    """x = softmax(lm_head(h) / tau) @ E -- the expected token embedding
    (MUX: skip argmax / sampling, feed the whole distribution). No
    parameters; gradients flow through the softmax into the trunk and
    the tied embedding. tau defaults to Cfg.tau. top_k keeps the k most
    likely tokens (renormalized) -- a sparse superposition.
    """

    def __init__(self, tau: float | None = None, top_k: int | None = None):
        super().__init__()
        self.tau, self.top_k = tau, top_k

    def probs(self, h, model):
        tau = self.tau if self.tau is not None else model.cfg.tau
        logits = model.lm_head(h).float() / tau
        if self.top_k:
            kth = logits.topk(int(self.top_k), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return F.softmax(logits, dim=-1)

    def forward(self, h, model):
        p = self.probs(h, model)
        return (p @ model.tok_emb.weight.float()).to(h.dtype)


FEEDBACKS = {"projected": Projected, "expected": Expected}


def make_feedback(name: str, d_model: int, **opts) -> nn.Module:
    if name == "projected":
        return Projected(d_model, **opts)
    if name in FEEDBACKS:
        return FEEDBACKS[name](**opts)
    raise ValueError(f"unknown feedback '{name}' (one of: {', '.join(FEEDBACKS)})")
