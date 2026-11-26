"""The engine's `training_loss` hook, end to end with LatentLM's MuxLM
(stochastic depth + truncated BPTT) on tasks/program_trace.py."""

import csv

import pytest

import train
from conftest import run_mode, write_config

MUX = {
    "factory": "models/latent_lm.py:MuxLM",
    "attn_impl": "eager", "d_model": 32, "n_heads": 4, "d_ff": 64, "n_layers": 2,
    "d_rel": 8, "rel_extent": 64, "feedback": "expected", "n_latents": 3, "max_span": 8,
    "loop_sampler": "uniform", "loop_min": 1, "loop_max": 5, "loop_bptt": 2, "eval_loops": [1, 3],
}
TRACE = {"factory": "tasks/program_trace.py:ProgramTrace", "steps": 3, "val_size": 8}


def mux_overrides(**train_kw):
    return {"model": dict(MUX), "task": dict(TRACE),
            "train": {"epochs": 2, "batches_per_epoch": 3, "batch_size": 4, "lr": 1e-3, **train_kw}}


def test_pretrain_mode_logs_model_parts(tmp_path):
    rows = run_mode("pretrain", tmp_path / "mux", mux_overrides())
    cols = list(rows[0])
    for c in ("val_answer", "val_local", "val_answer_acc", "val_loops", "val_acc_k1", "val_acc_k3",
              "train_answer", "train_loops", "peak_mem_mb"):
        assert c in cols, c
    assert all(r["val_loops"] == "3.0" for r in rows), "eval uses n_latents"
    train_loops = [float(r["train_loops"]) for r in rows]
    assert all(1.0 <= k <= 5.0 for k in train_loops)
    assert len(set(train_loops)) > 1 or train_loops[0] != 3.0, "training samples K"


def test_runs_are_reproducible_with_sampled_depth(tmp_path):
    a = run_mode("pretrain", tmp_path / "a", mux_overrides())
    b = run_mode("pretrain", tmp_path / "b", mux_overrides())
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall"} for r in rows]
    assert strip(a) == strip(b)


def test_check_reports_the_hook(tmp_path):
    report = train.check(write_config(tmp_path / "chk", mux_overrides()))
    assert "training_loss hook" in report and "train step:" in report and "loops" in report


def test_ga_uses_the_hook_as_fitness(tmp_path):
    ov = mux_overrides()
    ov["model"].update(d_model=16, n_heads=2, d_ff=16, n_layers=1)
    ov["train"].update(pop_size=4, generations=2, checkpoint_every=1)
    rows = run_mode("ga", tmp_path / "ga", ov)
    assert len(rows) == 2 and float(rows[1]["gen_best"]) <= float(rows[0]["gen_best"])


@pytest.mark.parametrize("mode, needle", [("rl", "rl needs policy logits"), ("distill", "distill needs model outputs")])
def test_modes_that_need_outputs_refuse_clearly(tmp_path, mode, needle):
    ov = mux_overrides()
    if mode == "distill":
        run_mode("pretrain", tmp_path / "teacher", mux_overrides())
        ov["distill"] = {"teacher_checkpoint": str(tmp_path / "teacher/checkpoints/latest.pt")}
    config = write_config(tmp_path / mode, ov)
    assert train.execute(config, mode=mode) == 1
    assert needle in (tmp_path / mode / "FAILED").read_text()


def test_program_trace_ops_and_modulus(tmp_path):
    import sys
    sys.path.insert(0, str(train.EXPERIMENTS / "tasks"))
    import codetok
    from program_trace import ProgramTrace

    t = ProgramTrace(seed=0, steps=3, val_size=64, modulus=10, ops="+-")
    q = [codetok.decode(r) for r in t.val[0]["q"]]
    assert all("*" not in s for s in q) and any("+" in s for s in q) and any("-" in s for s in q)
    assert {int(codetok.decode(a[:1])) for a in t.val[0]["answer"]} <= set(range(10))
    with pytest.raises(ValueError, match="subset"):
        ProgramTrace(seed=0, ops="/")


def test_parts_include_the_teacher_and_schedule(tmp_path):
    ov = mux_overrides()
    ov["model"].update(gamma=2.0, gamma_warmup=3, global_loss="relative")
    rows = run_mode("pretrain", tmp_path / "g", ov)
    assert "val_ref_answer_acc" in rows[0] and "train_gamma" in rows[0]
    # 2 epochs x 3 calls, warmup 3, no ramp: epoch 1 all 0, epoch 2 all 2.0
    assert [float(r["train_gamma"]) for r in rows] == [0.0, 2.0]


def test_check_validates_the_monitor_section(tmp_path):
    ov = mux_overrides()
    ov["monitor"] = {"watch": [{"metric": "val_acc_k1", "goal": "max"}, {"metric": "val_acc_k9", "goal": "max"},
                               {"metric": "val_ref", "goal": "sideways"}],
                     "show": ["val_ref", "val_nope"]}
    report = train.check(write_config(tmp_path / "mon", ov))
    assert "monitor: val_acc_k1 (max), val_acc_k9 (max)" in report
    assert "monitor watches 'val_acc_k9', which this run does not log" in report
    assert "monitor config: watch[2] val_ref: goal must be min or max" in report
    assert "monitor.show lists 'val_nope'" in report
    assert "monitor watches 'val_acc_k1'" not in report


def test_discrete_objective_is_muxs_ref_term(tmp_path):
    import sys
    import torch
    sys.path[:0] = [str(train.EXPERIMENTS / "models"), str(train.EXPERIMENTS / "tasks")]
    from latent_lm import MuxLM
    from program_trace import ProgramTrace

    kw = {k: v for k, v in MUX.items() if k != "factory"}
    x, _ = ProgramTrace(seed=0, steps=3, val_size=4).val
    torch.manual_seed(0)
    a = MuxLM(**kw, objective="discrete").eval()
    b = MuxLM(**{**kw, "loop_sampler": "fixed"}).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        loss_a, parts_a = a.training_loss(x)
        _, parts_b = b.mux_loss(x["q"], None, None, x["trace"], x["answer"])
    assert loss_a.item() == pytest.approx(parts_b["ref"], rel=1e-5)
    assert parts_a["ref_answer_acc"] == parts_b["ref_answer_acc"]
    assert 0.0 <= parts_a["trace_acc"] <= 1.0
    with pytest.raises(ValueError, match="unknown objective"):
        MuxLM(**kw, objective="nope")


def test_trace_mask_marks_the_value_tokens():
    import sys
    sys.path.insert(0, str(train.EXPERIMENTS / "tasks"))
    import codetok
    from program_trace import ProgramTrace

    x, _ = ProgramTrace(seed=0, steps=3, val_size=8).val
    for row, mask in zip(x["trace"], x["trace_mask"]):
        picked = [codetok.decode([t]).strip() for t in row[mask]]
        assert len(picked) == 3 and all(p.isdigit() for p in picked)
        assert all(not codetok.decode([t]).strip().isdigit() for t in row[~mask])


def test_no_empty_train_columns_and_stage_columns(tmp_path):
    rows = run_mode("pretrain", tmp_path / "mux", mux_overrides())          # mux objective
    cols = list(rows[0])
    assert "val_acc_k1" in cols and "val_trace_acc" in cols
    assert not [c for c in cols if c.startswith("train_acc_k") or c == "train_trace_acc"]
    assert all(v != "" for r in rows for v in r.values()), "every logged cell is filled"
    ov = mux_overrides()
    ov["model"]["objective"] = "discrete"
    rows = run_mode("pretrain", tmp_path / "disc", ov)                     # stage 1
    assert {"train_ref", "train_trace_acc", "val_trace_acc", "val_ref_answer_acc"} <= set(rows[0])
    assert all(v != "" for r in rows for v in r.values())
    # stage 2 warm-starts from the stage-1 checkpoint (same parameter names)
    ov2 = mux_overrides()
    ov2["init"] = {"checkpoint": str(tmp_path / "disc/checkpoints/latest.pt")}
    assert "missing 0" in train.check(write_config(tmp_path / "stage2", ov2))
