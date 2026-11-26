"""Test-only model plug-in: the harness itself knows no architecture.

Flattens a (B, T, 1) bit sequence and maps it to (B, n_classes) logits.
"""

import torch


class MLP(torch.nn.Module):
    def __init__(self, seq_len: int = 6, hidden: int = 8, n_classes: int = 2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(seq_len, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def broken(**_):
    """A factory that returns something that is not a module."""
    return object()


class TinyLM(torch.nn.Module):
    """ids (B, T) -> logits (B, T, vocab): embedding + MLP, no attention."""

    def __init__(self, vocab_size: int = 16, d: int = 8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, d)
        self.out = torch.nn.Linear(d, vocab_size)

    def forward(self, ids):
        return self.out(torch.tanh(self.emb(ids)))
