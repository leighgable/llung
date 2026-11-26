"""Engine features for LM-style runs: the `tokens` task, warm start
(init + convert), lr schedule, grad accumulation, chunked eval,
checkpoint pruning, and source snapshots of multi-file plug-ins."""

import json

import numpy as np
import pytest

import train
from conftest import TESTS, run_mode, strip_wall, write_config

VOCAB = 16


def make_corpus(path, n=4000, seed=0):
    """A learnable stream: token t+1 = (t + 1) % VOCAB, with noise."""
    rng = np.random.default_rng(seed)
    toks = (np.arange(n) % VOCAB).astype(np.uint16)
    noise = rng.random(n) < 0.1
    toks[noise] = rng.integers(0, VOCAB, noise.sum())
    path.mkdir(parents=True, exist_ok=True)
    toks.tofile(path / "train.bin")
    toks[: n // 4].tofile(path / "val.bin")
    (path / "meta.json").write_text(json.dumps({"dtype": "uint16", "vocab_size": VOCAB}))
    return path


def lm_overrides(data, **train_kw):
    return {
        "task": {"factory": "tokens", "path": str(data), "seq_len": 12, "val_size": 20},
        "model": {"factory": f"{TESTS / 'toy_models.py'}:TinyLM", "vocab_size": VOCAB, "d": 8},
        "train": {"epochs": 3, "batches_per_epoch": 4, "batch_size": 4, "lr": 0.05,
                  "eval_batch_size": 7, **train_kw},
    }


def test_tokens_task_trains_with_accum_schedule_and_amp(tmp_path):
    data = make_corpus(tmp_path / "data")
    ov = lm_overrides(data, grad_accum=2, lr_schedule="cosine", warmup_steps=3, amp="bf16")
    rows = run_mode("pretrain", tmp_path / "lm", ov)
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]
    assert "val_acc" in rows[0]
    assert float(rows[-1]["val_loss"]) < float(rows[0]["val_loss"])
    # deterministic, including the windows sampled from the memmap
    again = run_mode("pretrain", tmp_path / "lm2", ov)
    assert strip_wall(rows) == strip_wall(again)


def test_lr_schedule():
    tr = {"lr": 1.0, "warmup_steps": 4, "lr_schedule": "cosine", "min_lr_frac": 0.1}
    assert train.lr_at(0, 100, tr) == pytest.approx(0.25)
    assert train.lr_at(3, 100, tr) == pytest.approx(1.0)
    assert train.lr_at(4, 100, tr) == pytest.approx(1.0)
    assert train.lr_at(100, 100, tr) == pytest.approx(0.1)
    assert train.lr_at(50, 100, {"lr": 0.3}) == 0.3


def test_chunked_eval_matches_one_chunk(tmp_path):
    data = make_corpus(tmp_path / "data")
    cfg = train.load_config(write_config(tmp_path / "a", lm_overrides(data, eval_batch_size=3)))
    exp = train.Experiment(cfg, train.search_path(tmp_path / "a" / "config.yaml"), "cpu")
    chunked = exp.evaluate()
    cfg["train"]["eval_batch_size"] = 1000
    whole = exp.evaluate()
    assert chunked[0] == pytest.approx(whole[0], rel=1e-5)
    assert chunked[1]["val_acc"] == pytest.approx(whole[1]["val_acc"], abs=1e-5)


def test_keep_checkpoints(tmp_path):
    run_mode("pretrain", tmp_path / "k", {"train": {"epochs": 5, "checkpoint_every": 1, "keep_checkpoints": 2}})
    names = sorted(p.name for p in (tmp_path / "k" / "checkpoints").glob("step_*.pt"))
    assert names == ["step_4.pt", "step_5.pt"]


def test_warm_start_from_a_previous_run(tmp_path):
    run_mode("pretrain", tmp_path / "base", {"train": {"epochs": 4}})
    ckpt = tmp_path / "base" / "checkpoints" / "latest.pt"
    config = write_config(tmp_path / "warm", {"init": {"checkpoint": str(ckpt)}})
    report = train.check(config)
    assert "via load_state_dict(matching keys)" in report and "missing 0" in report
    # identical weights -> identical val loss as the base run's final eval
    cfg = train.load_config(config)
    exp = train.Experiment(cfg, train.search_path(config), "cpu")
    import torch
    base = torch.load(ckpt, weights_only=False)["model"]
    assert all(torch.equal(base[k], v) for k, v in exp.model.state_dict().items())


def test_warm_start_with_converter_and_mismatch_report(tmp_path):
    run_mode("pretrain", tmp_path / "base", {"model": {"hidden": 16}})
    run = tmp_path / "conv"
    run.mkdir()
    # a "foreign" backbone: rename every key with a prefix the converter strips
    (run / "conv.py").write_text(
        "def convert(sd):\n"
        "    return {k.removeprefix('backbone.'): v for k, v in sd.items()}\n"
    )
    import torch
    sd = torch.load(tmp_path / "base/checkpoints/latest.pt", weights_only=False)["model"]
    torch.save({f"backbone.{k}": v for k, v in sd.items()}, run / "foreign.bin")
    config = write_config(run, {"init": {"checkpoint": "foreign.bin", "convert": "conv.py:convert"}})
    report = train.check(config)  # hidden=8 model vs hidden=16 weights
    assert "shape-mismatch" in report and "shape-mismatch 0" not in report
    assert "unexpected 0" in report
    # strict: the same mismatch is a hard failure -> FAILED
    write_config(run, {"init": {"checkpoint": "foreign.bin", "convert": "conv.py:convert", "strict": True}})
    assert train.execute(run / "config.yaml") == 1
    assert "size mismatch" in (run / "FAILED").read_text()


def test_model_warm_start_from_hook_is_used(tmp_path):
    run = tmp_path / "hook"
    run.mkdir()
    (run / "hooked.py").write_text(
        "import sys; sys.path.insert(0, %r)\n"
        "from toy_models import MLP\n"
        "class Hooked(MLP):\n"
        "    calls = 0\n"
        "    def warm_start_from(self, sd):\n"
        "        type(self).calls += 1\n"
        "        self.load_state_dict(sd)\n" % str(TESTS)
    )
    run_mode("pretrain", tmp_path / "base")
    config = write_config(run, {"model": {"factory": "hooked.py:Hooked", "seq_len": 6, "hidden": 8},
                                "init": {"checkpoint": str(tmp_path / "base/checkpoints/latest.pt")}})
    assert "via model.warm_start_from" in train.check(config)


def test_sources_snapshot_includes_sibling_modules(tmp_path):
    run = tmp_path / "sib"
    run.mkdir()
    (run / "arch.py").write_text((TESTS / "toy_models.py").read_text())
    (run / "adapter.py").write_text("from arch import MLP\n\nclass Adapter(MLP):\n    pass\n")
    run_mode("pretrain", run, {"model": {"factory": "adapter.py:Adapter", "seq_len": 6, "hidden": 8}})
    assert {"adapter.py", "arch.py"} <= {p.name for p in (run / "src").iterdir()}
    # edits to the sibling are picked up in the same process (no stale
    # module cache; the size change also defeats the 1 s pyc mtime check)
    (run / "arch.py").write_text((TESTS / "toy_models.py").read_text().replace("torch.nn.Tanh()", "torch.nn.ReLU() "))
    cfg = train.load_config(run / "config.yaml")
    exp = train.Experiment(cfg, train.search_path(run / "config.yaml"), "cpu")
    assert any(type(m).__name__ == "ReLU" for m in exp.model.modules())
