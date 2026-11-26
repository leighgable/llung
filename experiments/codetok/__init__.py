"""codetok -- THE tokenizer for the code/reasoning model experiments.

CodeParrot's byte-level BPE (codeparrot/codeparrot-small, 32768 tokens,
pinned in build.py) plus 128 special-token slots at ids 32768..32895:

    vocab 32896 = 32768 BPE + 128 special   (a multiple of 128)

    EOS        0      <|endoftext|>   document separator / end of sample
    PAD    32768      <|pad|>         padding: never a loss target
    BOT    32769      <|bot|>         begin latent thought (LatentLM bot_id)
    EOT    32770      <|eot|>         end latent thought   (LatentLM eot_id)
    IM_START / IM_END                 chat turns (ChatML; role name as text)
    THINK / END_THINK                 discrete reasoning trace <think> ... </think>
    FIM_PREFIX / FIM_MIDDLE / FIM_SUFFIX   code infilling
    TOOL_CALL / TOOL_RESULT           code execution / tools (RL with test feedback)
    <|reserved_N|>                    free slots: rename one, rebuild; the vocab
                                      size (and every checkpoint) stays valid

Base ids are identical to upstream, so the upstream model's embeddings
warm-start rows 0..32767 directly (models/gpt2_to_latentlm.py fills the
128 new rows).

Text never turns into special tokens: `encode` uses the BPE WITHOUT the
added-token matcher, so a source file that literally contains "<think>"
or "<|endoftext|>" stays text. Special ids are inserted structurally
(`chat`, or by hand from the constants below).

    import codetok
    ids = codetok.encode("def f(x):\\n    return x")
    codetok.decode(ids)
    ids, mask = codetok.chat([{"role": "user", "content": "..."},
                              {"role": "assistant", "content": "...", "think": "..."}])
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().with_name("tokenizer.json")
__snapshot__ = [PATH]  # the engine copies this into each run's src/

BASE_VOCAB = 32768
_NAMED = [
    "<|pad|>",
    "<|bot|>",
    "<|eot|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|tool_call|>",
    "<|tool_result|>",
]
N_SPECIAL = 128
SPECIALS = _NAMED + [f"<|reserved_{i}|>" for i in range(N_SPECIAL - len(_NAMED))]
VOCAB_SIZE = BASE_VOCAB + N_SPECIAL

EOS = 0
PAD, BOT, EOT, IM_START, IM_END, THINK, END_THINK, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, TOOL_CALL, TOOL_RESULT = (
    BASE_VOCAB + i for i in range(len(_NAMED))
)
ROLES = ("system", "user", "assistant", "tool")


@lru_cache(maxsize=None)
def _full():
    from tokenizers import Tokenizer

    if not PATH.exists():
        raise FileNotFoundError(f"{PATH} missing -- run: python experiments/codetok/build.py")
    return Tokenizer.from_file(str(PATH))


@lru_cache(maxsize=None)
def _text():
    """The same BPE with the added-token matcher removed: literal
    special-token strings in text stay ordinary text."""
    from tokenizers import Tokenizer

    spec = json.loads(PATH.read_text())
    spec["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(spec))


def load(fresh: bool = False):
    """The full `tokenizers.Tokenizer` (specials included)."""
    if fresh:
        _full.cache_clear()
        _text.cache_clear()
    return _full()


def sha256() -> str:
    return hashlib.sha256(PATH.read_bytes()).hexdigest()


def encode(text: str) -> list[int]:
    """Plain text -> BPE ids (all < BASE_VOCAB; never a special id)."""
    return _text().encode(text, add_special_tokens=False).ids


def encode_batch(texts: list[str]) -> list[list[int]]:
    return [e.ids for e in _text().encode_batch(texts, add_special_tokens=False)]


def decode(ids, keep_special: bool = True) -> str:
    """ids -> text. Special tokens are shown by name unless keep_special
    is False; PAD is always dropped."""
    ids = [int(i) for i in ids if int(i) != PAD]
    return _full().decode(ids, skip_special_tokens=not keep_special)


def token_id(name: str) -> int:
    i = _full().token_to_id(name)
    if i is None:
        raise KeyError(name)
    return i


def chat(messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[bool]]:
    """ChatML ids + a loss mask (True = train on this token).

        <|im_start|>{role}\\n{content}<|im_end|>\\n

    An assistant message may carry "think" (a discrete reasoning trace),
    rendered as <think>{think}</think> before the content. The mask
    covers what the model must produce: assistant think + content +
    <|im_end|>. add_generation_prompt appends "<|im_start|>assistant\\n"
    (for sampling).
    """
    ids: list[int] = []
    mask: list[bool] = []

    def put(toks, train):
        ids.extend(toks)
        mask.extend([train] * len(toks))

    for m in messages:
        role = m["role"]
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r} (expected one of {ROLES})")
        train = role == "assistant"
        put([IM_START] + encode(f"{role}\n"), False)
        if m.get("think") is not None:
            if not train:
                raise ValueError("only assistant messages carry 'think'")
            put([THINK] + encode(m["think"]) + [END_THINK], True)
        put(encode(m["content"]) + [IM_END], train)
        put(encode("\n"), False)
    if add_generation_prompt:
        put([IM_START] + encode("assistant\n"), False)
    return ids, mask
