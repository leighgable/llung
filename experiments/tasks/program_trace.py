"""tasks/program_trace.py -- a toy CODE-REASONING task for the latent loop.

A straight-line program over one variable, mod 100; the model must
"execute" it. Question, trace and answer are codetok ids:

    q       x = 42\\nx = x * 7\\nx = x + 3\\nprint(x)\\n<|bot|>
    trace   x = 94\\nx = 97\\n                 (the state after every statement)
    answer  97<|endoftext|>

Every value 0..99 is a single codetok token, so all rows have the same
length (LatentLM has no attention padding mask yet): q = 4 + 6*steps + 5
+ 1 tokens, trace = 4*steps, answer = 2.

Use with models/latent_lm.py:MuxLM (batches are dicts {q, trace, answer,
trace_mask}; y = answer). trace_mask marks the value token of every
trace line -- the only trace tokens that need computing ("x", " =" and
"\n" are free), so MuxLM's trace_acc measures arithmetic, not syntax. The trace has `steps` lines of 4 tokens: with K = steps
latent steps, the chunked MUX spans are exactly one statement each.

    task:
      factory: tasks/program_trace.py:ProgramTrace
      steps: 4
      ops: "+-*"        # a curriculum knob: "+-" and a small modulus are much easier
      modulus: 100
"""

from __future__ import annotations

import numpy as np
import torch

import codetok

OPS = ("+", "-", "*")


def apply(op: str, v: int, d: int, modulus: int) -> int:
    return {"+": v + d, "-": v - d, "*": v * d}[op] % modulus


class ProgramTrace:
    def __init__(self, seed: int, steps: int = 4, val_size: int = 256, modulus: int = 100, ops: str = "+-*"):
        if not 1 <= modulus <= 100:
            raise ValueError("modulus must be in 1..100 (values are single tokens)")
        if not ops or set(ops) - set(OPS):
            raise ValueError(f"ops must be a non-empty subset of {''.join(OPS)!r}")
        self.steps, self.modulus, self.ops = int(steps), int(modulus), tuple(dict.fromkeys(ops))
        enc = codetok.encode
        self._init = [enc(f"x = {v}\n") for v in range(100)]
        self._stmt = {(op, d): enc(f"x = x {op} {d}\n") for op in OPS for d in range(1, 10)}
        self._val = [enc(f"{v}") for v in range(100)]
        self._tail = enc("print(x)\n") + [codetok.BOT]
        for table, n in ((self._init, 4), (self._stmt.values(), 6), (self._val, 1)):
            assert all(len(t) == n for t in table), "fixed-length rows need fixed-length statements"
        self._seed = int(seed) + 2
        self.val = self._batch(int(val_size), np.random.default_rng(int(seed) + 1))

    def _row(self, rng: np.random.Generator):
        v = int(rng.integers(self.modulus))
        q, trace = list(self._init[v]), []
        for _ in range(self.steps):
            op, d = self.ops[int(rng.integers(len(self.ops)))], int(rng.integers(1, 10))
            v = apply(op, v, d, self.modulus)
            q += self._stmt[(op, d)]
            trace += self._init[v]
        return q + self._tail, trace, self._val[v] + [codetok.EOS]

    def _batch(self, n: int, rng: np.random.Generator):
        rows = [self._row(rng) for _ in range(n)]
        q, trace, answer = (torch.tensor([r[i] for r in rows]) for i in range(3))
        mask = torch.zeros_like(trace, dtype=torch.bool)
        mask[:, 2::4] = True  # "x", " =", VALUE, "\n" per line
        return {"q": q, "trace": trace, "answer": answer, "trace_mask": mask}, answer

    def sample(self, n: int, step: int):
        return self._batch(n, np.random.default_rng([self._seed, step]))
