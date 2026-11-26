"""codetok (experiments/codetok): the shared tokenizer for the code /
reasoning model experiments, and its integration with the engine."""

import json
import sys

import pytest
import torch
from hypothesis import given, settings, strategies as st

import train
from conftest import run_mode, write_config

sys.path.insert(0, str(train.EXPERIMENTS))
import codetok  # noqa: E402

PROPS = {"deadline": None, "derandomize": True}
SPECIAL_STRINGS = st.sampled_from(["<|endoftext|>", *codetok.SPECIALS[:16]])


def test_vocab_layout_is_stable():
    tok = codetok.load()
    assert tok.get_vocab_size() == codetok.VOCAB_SIZE == 32896 and codetok.VOCAB_SIZE % 128 == 0
    assert tok.id_to_token(codetok.EOS) == "<|endoftext|>"
    named = {"PAD": "<|pad|>", "BOT": "<|bot|>", "EOT": "<|eot|>", "IM_START": "<|im_start|>",
             "IM_END": "<|im_end|>", "THINK": "<think>", "END_THINK": "</think>",
             "FIM_PREFIX": "<|fim_prefix|>", "FIM_MIDDLE": "<|fim_middle|>", "FIM_SUFFIX": "<|fim_suffix|>",
             "TOOL_CALL": "<|tool_call|>", "TOOL_RESULT": "<|tool_result|>"}
    for const, name in named.items():
        assert tok.token_to_id(name) == getattr(codetok, const), const
    for i, name in enumerate(codetok.SPECIALS):
        assert tok.token_to_id(name) == codetok.BASE_VOCAB + i
    # LatentLM's own defaults (1, 2, 0) are ordinary tokens / EOS here.
    assert len({codetok.BOT, codetok.EOT, codetok.PAD, codetok.EOS}) == 4


@given(st.text(max_size=300))
@settings(max_examples=150, **PROPS)
def test_roundtrip_and_text_never_yields_special_ids(text):
    ids = codetok.encode(text)
    assert codetok.decode(ids) == text
    assert all(0 < i < codetok.BASE_VOCAB for i in ids)


@given(st.lists(st.one_of(st.text(max_size=20), SPECIAL_STRINGS), max_size=12))
@settings(max_examples=100, **PROPS)
def test_literal_special_strings_stay_text(parts):
    """Code that literally contains "<think>" or "<|endoftext|>" must
    not be tokenized into control tokens."""
    text = "".join(parts)
    ids = codetok.encode(text)
    assert all(0 < i < codetok.BASE_VOCAB for i in ids)
    assert codetok.decode(ids) == text


def test_chat_template_and_loss_mask():
    msgs = [
        {"role": "system", "content": "You write Python."},
        {"role": "user", "content": "square x"},
        {"role": "assistant", "content": "x * x", "think": "multiply"},
    ]
    ids, mask = codetok.chat(msgs)
    assert len(ids) == len(mask)
    assert codetok.decode(ids) == (
        "<|im_start|>system\nYou write Python.<|im_end|>\n"
        "<|im_start|>user\nsquare x<|im_end|>\n"
        "<|im_start|>assistant\n<think>multiply</think>x * x<|im_end|>\n"
    )
    trained = [i for i, m in zip(ids, mask) if m]
    assert codetok.decode(trained) == "<think>multiply</think>x * x<|im_end|>"
    prompt, pmask = codetok.chat(msgs[:2], add_generation_prompt=True)
    assert codetok.decode(prompt).endswith("<|im_start|>assistant\n") and not any(pmask)
    with pytest.raises(ValueError):
        codetok.chat([{"role": "user", "content": "x", "think": "no"}])
    with pytest.raises(ValueError):
        codetok.chat([{"role": "robot", "content": "x"}])


def test_decode_drops_pad_and_can_hide_specials():
    ids = [codetok.PAD, codetok.THINK, *codetok.encode("a"), codetok.END_THINK, codetok.PAD]
    assert codetok.decode(ids) == "<think>a</think>"
    assert codetok.decode(ids, keep_special=False) == "a"


def test_converter_extends_the_vocab_with_mean_rows():
    sys.path.insert(0, str(train.EXPERIMENTS / "models"))
    from gpt2_to_latentlm import extend_vocab

    emb = torch.randn(codetok.BASE_VOCAB, 4)
    out = extend_vocab(emb, codetok.VOCAB_SIZE)
    assert out.shape == (codetok.VOCAB_SIZE, 4)
    assert torch.equal(out[: codetok.BASE_VOCAB], emb)
    assert torch.allclose(out[codetok.BASE_VOCAB :], emb.mean(0).expand(codetok.N_SPECIAL, -1))


def test_runs_snapshot_the_tokenizer(tmp_path):
    """A plug-in importing codetok -> the run's src/ carries the package
    AND its tokenizer.json (the tokenizer is part of the provenance)."""
    run = tmp_path / "tok"
    run.mkdir()
    (run / "uses_tok.py").write_text(
        "import codetok, sys\n"
        f"sys.path.insert(0, {str(train.CORE / 'tests')!r})\n"
        "from toy_models import MLP\n"
        "class M(MLP):\n"
        "    vocab = codetok.VOCAB_SIZE\n"
    )
    run_mode("pretrain", run, {"model": {"factory": "uses_tok.py:M", "seq_len": 6, "hidden": 8}})
    snap = run / "src" / "codetok" / "tokenizer.json"
    assert snap.exists() and snap.read_bytes() == codetok.PATH.read_bytes()


def test_check_warns_when_the_model_vocab_is_smaller_than_the_data(tmp_path):
    from test_lm import make_corpus, lm_overrides

    data = make_corpus(tmp_path / "data")
    meta = json.loads((data / "meta.json").read_text())
    meta.update(vocab_size=32, tokenizer="codetok", tokenizer_sha256="abc")
    (data / "meta.json").write_text(json.dumps(meta))
    report = train.check(write_config(tmp_path / "small", lm_overrides(data)))  # model vocab 16
    assert "data:    tokenizer codetok" in report
    assert "model outputs 16 logits but the task's vocab is 32" in report
