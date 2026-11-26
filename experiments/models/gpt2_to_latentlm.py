"""Rename a GPT-2-family state_dict (e.g. codeparrot/codeparrot-small's
pytorch_model.bin) into LatentLM's parameter names.

Use as the engine's init converter:

    init:
      checkpoint: <data dir>/codeparrot-small.bin
      convert: models/gpt2_to_latentlm.py:convert

GPT-2 -> LatentLM, per layer i:
    ln_1 / ln_2                 -> blocks.i.ln1 / ln2          (exact)
    attn.c_attn.weight^T        -> blocks.i.qkv.weight         (same q|k|v order and head split)
    attn.c_proj.weight^T        -> blocks.i.o_proj.weight
    mlp.c_fc / mlp.c_proj (^T)  -> blocks.i.ff.0 / ff.2        (weights + biases)
    wte / ln_f                  -> tok_emb / final_ln          (lm_head is tied to tok_emb)

NOT carried over (LatentLM has no slot for them):
    wpe                 absolute positions -- LatentLM uses the learned
                        relative bias, which starts at zero (warm_start)
    attn.c_attn.bias    qkv has bias=False; the k bias cancels in softmax,
                        the q/v biases are lost
    attn.c_proj.bias    o_proj has bias=False
The vocab grows from GPT-2's 32768 to codetok.VOCAB_SIZE (32896: the
128 special-token slots). Rows 0..32767 are copied (codetok keeps the
base ids); the new rows start at the MEAN embedding, so the tied LM
head gives new tokens an average logit instead of a random one.

so step 0 is NOT the backbone's loss: expect it to start well above
codeparrot-small and recover as the relative bias and the attention
projections adapt. GPT-2's MLP uses tanh-approx GELU ("gelu_new");
LatentLM's nn.GELU() is exact (a negligible difference).

GPT-2 Conv1D stores weights as (in, out); nn.Linear wants (out, in),
hence the transposes.
"""

import re

import codetok
import torch

_LAYER = re.compile(r"^h\.(\d+)\.(.+)$")
_PER_LAYER = {
    "ln_1.weight": ("ln1.weight", False),
    "ln_1.bias": ("ln1.bias", False),
    "ln_2.weight": ("ln2.weight", False),
    "ln_2.bias": ("ln2.bias", False),
    "attn.c_attn.weight": ("qkv.weight", True),
    "attn.c_proj.weight": ("o_proj.weight", True),
    "mlp.c_fc.weight": ("ff.0.weight", True),
    "mlp.c_fc.bias": ("ff.0.bias", False),
    "mlp.c_proj.weight": ("ff.2.weight", True),
    "mlp.c_proj.bias": ("ff.2.bias", False),
}
_TOP = {"wte.weight": "tok_emb.weight", "ln_f.weight": "final_ln.weight", "ln_f.bias": "final_ln.bias"}


def extend_vocab(emb: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Append rows (mean-initialized) up to vocab_size."""
    extra = vocab_size - emb.shape[0]
    if extra < 0:
        raise ValueError(f"backbone vocab {emb.shape[0]} > target {vocab_size}")
    return torch.cat([emb, emb.mean(0, keepdim=True).expand(extra, -1)]) if extra else emb


def convert(sd: dict, vocab_size: int = codetok.VOCAB_SIZE) -> dict:
    out, dropped = {}, []
    for key, value in sd.items():
        name = key.removeprefix("transformer.")
        if name in _TOP:
            out[_TOP[name]] = value
            continue
        m = _LAYER.match(name)
        if m and m.group(2) in _PER_LAYER:
            target, transpose = _PER_LAYER[m.group(2)]
            out[f"blocks.{m.group(1)}.{target}"] = value.t().contiguous() if transpose else value
        else:
            dropped.append(key)
    if "tok_emb.weight" not in out:
        raise KeyError("no wte.weight found -- is this a GPT-2-family state_dict?")
    base = out["tok_emb.weight"].shape[0]
    out["tok_emb.weight"] = extend_vocab(out["tok_emb.weight"], vocab_size)
    kinds = sorted({re.sub(r"^(transformer\.)?h\.\d+\.", "h.*.", k) for k in dropped})
    print(f"gpt2_to_latentlm: converted {len(out)} tensors, dropped {len(dropped)}: {', '.join(kinds)}; "
          f"vocab {base} -> {vocab_size} ({vocab_size - base} new rows, mean-initialized)")
    return out


if __name__ == "__main__":  # quick look: python gpt2_to_latentlm.py <pytorch_model.bin>
    import sys

    converted = convert(torch.load(sys.argv[1], map_location="cpu", weights_only=True))
    for k, v in list(converted.items())[:14]:
        print(f"  {k:28s} {list(v.shape)}")
