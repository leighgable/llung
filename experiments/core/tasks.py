"""core/tasks.py -- built-in toy tasks (data only, no models).

A task is any object with this shape (see train.py "Plug-in contract"):

    val                          fixed validation batch (x, y)
    sample(n, step) -> (x, y)    deterministic in `step`
    loss(out, y)                 optional; default by target dtype
    metrics(out, y) -> dict      optional; extra metrics.csv columns
    reward(x, actions, y)        optional; rl mode (default: actions == y)

parity / lastbit are sequence classification over random bits:

    x: float (batch, seq_len, 1)     y: long (batch,) in {0, 1}

so a model for them maps (B, T, 1) -> logits (B, 2). Use them as the
smoke test for a new architecture.

tokens is next-token prediction over a pre-tokenized corpus:

    x: long (batch, seq_len)         y: long (batch, seq_len)  (x shifted by 1)

so a model maps (B, T) ids -> logits (B, T, vocab). Write your own task
for anything else and point `task.factory` at it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class BitTask:
    n_classes = 2

    def __init__(self, seed: int, seq_len: int = 8, val_size: int = 256):
        self.seq_len = int(seq_len)
        self._seed = int(seed) + 2
        g = torch.Generator().manual_seed(int(seed) + 1)
        self.val = self._batch(int(val_size), g)

    def target(self, bits: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B,)
        raise NotImplementedError

    def _batch(self, n: int, g: torch.Generator):
        bits = torch.randint(0, 2, (n, self.seq_len), generator=g)
        return bits.unsqueeze(-1).float(), self.target(bits).long()

    def sample(self, n: int, step: int):
        return self._batch(n, torch.Generator().manual_seed(self._seed + step))

    def metrics(self, out: torch.Tensor, y: torch.Tensor) -> dict:
        return {"val_acc": (out.argmax(-1) == y).float().mean().item()}


class Parity(BitTask):
    """target = XOR of the sequence (hard for shallow models)."""

    def target(self, bits):
        return bits.sum(dim=1).remainder(2)


class LastBit(BitTask):
    """target = final bit (sanity task: loss floor is 0)."""

    def target(self, bits):
        return bits[:, -1]


class TokenStream:
    """Language modelling over a pre-tokenized corpus (nanoGPT layout):

        <path>/train.bin   flat token ids (np.memmap)
        <path>/val.bin
        <path>/meta.json   {"dtype": "uint16", "vocab_size": ..., ...}

    Each example is a random window of seq_len + 1 tokens; `sample` is
    deterministic in (seed, step), the val windows are fixed. The files
    are opened read-only, so they can live in a read-only data dir.
    """

    def __init__(self, seed: int, path: str, seq_len: int = 256, val_size: int = 64):
        self.path = Path(path).expanduser()
        meta_file = self.path / "meta.json"
        self.meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        dtype = np.dtype(self.meta.get("dtype", "uint16"))
        self.train = np.memmap(self.path / "train.bin", dtype=dtype, mode="r")
        self.val_tokens = np.memmap(self.path / "val.bin", dtype=dtype, mode="r")
        self.vocab_size = self.meta.get("vocab_size")
        self.seq_len = int(seq_len)
        for name, data in (("train.bin", self.train), ("val.bin", self.val_tokens)):
            if len(data) <= self.seq_len + 1:
                raise ValueError(f"{self.path / name} has {len(data)} tokens; need > seq_len + 1")
        self._seed = int(seed) + 2
        self.val = self._windows(self.val_tokens, int(val_size), np.random.default_rng(int(seed) + 1))

    def _windows(self, data: np.ndarray, n: int, rng: np.random.Generator):
        starts = rng.integers(0, len(data) - self.seq_len - 1, size=n)
        chunk = torch.from_numpy(np.stack([data[s : s + self.seq_len + 1] for s in starts]).astype(np.int64))
        return chunk[:, :-1], chunk[:, 1:]

    def sample(self, n: int, step: int):
        return self._windows(self.train, n, np.random.default_rng([self._seed, step]))

    def metrics(self, out: torch.Tensor, y: torch.Tensor) -> dict:
        return {"val_acc": (out.argmax(-1) == y).float().mean().item()}
