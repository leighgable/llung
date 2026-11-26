"""LatentLM's latent loop: stochastic depth (sample_loops), span
re-chunking for a sampled K (chunk_spans), and truncated BPTT."""

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings, strategies as st

MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS))
sys.path.insert(0, str(MODELS.parent))

import transformer as T  # noqa: E402

PAD = 0
TINY = dict(vocab_size=97, d_model=32, n_heads=4, d_ff=64, n_layers=2, n_latents=4,
            max_span=4, d_rel=8, rel_extent=32, attn_impl="eager", gamma=1.0)
PROPS = {"deadline": None, "derandomize": True}


def model(**kw):
    torch.manual_seed(0)
    return T.LatentLM(T.Cfg(**{**TINY, **kw}), pad_id=PAD)


def batch(seed=1, lt=10):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(3, 97, (2, 5), generator=g)
    ans = torch.randint(3, 97, (2, 3), generator=g)
    trace = torch.randint(3, 97, (2, lt), generator=g)
    return q, ans, trace


# ------------------------------------------------------- stochastic depth


@pytest.mark.parametrize("kind", ["poisson", "lognormal_poisson", "uniform"])
def test_sampler_range_mean_and_determinism(kind):
    cfg = T.Cfg(**TINY, loop_sampler=kind, loop_mean=4, loop_min=1, loop_max=8)
    torch.manual_seed(0)
    draws = [T.sample_loops(cfg) for _ in range(4000)]
    assert min(draws) >= 1 and max(draws) <= 8
    counts = Counter(draws)
    assert len(counts) >= 6, f"{kind} should spread over the range: {counts}"
    mean = sum(draws) / len(draws)
    expected = 4.5 if kind == "uniform" else 4.0
    assert abs(mean - expected) < 0.35, (kind, mean)
    torch.manual_seed(0)
    assert [T.sample_loops(cfg) for _ in range(4000)] == draws, "seeded -> reproducible"


def test_lognormal_poisson_has_the_heavier_tail():
    torch.manual_seed(0)
    cfg = dict(TINY, loop_mean=4, loop_min=1, loop_max=64)
    tail = lambda kind: sum(T.sample_loops(T.Cfg(**cfg, loop_sampler=kind)) >= 10 for _ in range(4000))
    assert tail("lognormal_poisson") > 2 * tail("poisson")


def test_fixed_sampler_and_eval_mode_use_n_latents():
    m = model(loop_sampler="poisson", loop_mean=6)
    m.eval()
    assert {m.sample_n_latents() for _ in range(50)} == {4}
    m.train()
    assert len({m.sample_n_latents() for _ in range(200)}) > 1
    assert {T.sample_loops(T.Cfg(**TINY)) for _ in range(20)} == {4}
    with pytest.raises(ValueError, match="unknown loop_sampler"):
        T.sample_loops(T.Cfg(**TINY, loop_sampler="nope"))


# ----------------------------------------------------------- chunk_spans


@given(st.integers(1, 30), st.integers(0, 5), st.integers(1, 9), st.integers(1, 6))
@settings(max_examples=200, **PROPS)
def test_chunk_spans_partitions_the_trace(L, pad, K, S):
    """Spans are contiguous, in order, near-equal (sizes differ by <= 1
    before capping at max_span), masks are row prefixes, and the
    concatenation of the kept tokens is the trace minus what the caps cut."""
    trace = torch.cat([torch.arange(1, L + 1), torch.zeros(pad, dtype=torch.long)])[None]
    spans, mask, coverage = T.chunk_spans(trace, K, S, PAD)
    assert spans.shape == mask.shape == (1, K, S)
    # mask is a prefix of each row
    assert (mask.long().diff(dim=-1) <= 0).all()
    sizes = [((k * L + K - 1) // K, ((k + 1) * L + K - 1) // K) for k in range(K)]
    raw = [b - a for a, b in sizes]
    assert max(raw) - min(raw) <= 1 and sum(raw) == L
    for k, (a, b) in enumerate(sizes):
        n = min(b - a, S)
        assert mask[0, k].sum() == n
        assert spans[0, k, :n].tolist() == list(range(a + 1, a + 1 + n))
        assert (spans[0, k, n:] == PAD).all()
    assert coverage == pytest.approx(sum(min(r, S) for r in raw) / L)


def test_chunk_spans_recovers_statement_aligned_spans():
    """Equal-length trace lines and K = #lines -> one line per span."""
    lines = torch.arange(1, 13).view(3, 4)                       # 3 lines x 4 tokens
    spans, mask, cov = T.chunk_spans(lines.reshape(1, -1), 3, 4, PAD)
    assert torch.equal(spans[0], lines) and mask.all() and cov == 1.0


def test_mux_loss_derives_spans_for_any_K():
    q, ans, trace = batch()
    m = model()
    for K in (1, 3, 7):
        loss, parts = m.mux_loss(q, None, None, trace, ans, n_latents=K)
        assert torch.isfinite(loss) and "span_coverage" in parts
    loss, parts = m.mux_loss(q, None, None, trace, ans, n_latents=1)
    assert parts["span_coverage"] == pytest.approx(4 / 10), "K * max_span < L truncates the targets"


# ------------------------------------------------------- truncated BPTT


def grads(m):
    return {n: (p.grad.clone() if p.grad is not None else None) for n, p in m.named_parameters()}


@pytest.mark.parametrize("feedback", ["expected", "projected"])
def test_truncation_changes_gradients_not_values(feedback):
    q, ans, trace = batch()
    K = 6
    full = model(feedback=feedback)
    loss_full, parts_full = full.mux_loss(q, None, None, trace, ans, n_latents=K)
    loss_full.backward()
    g_full = grads(full)
    for window in (1, 2, 3):
        m = model(feedback=feedback)
        loss, parts = m.mux_loss(q, None, None, trace, ans, n_latents=K, bptt=window)
        assert torch.equal(loss, loss_full) and parts == parts_full, "forward values are identical"
        loss.backward()
        g = grads(m)
        assert all(torch.isfinite(v).all() for v in g.values() if v is not None)
        assert g["blocks.0.qkv.weight"].abs().sum() > 0, "the window still trains the trunk"
        assert any(not torch.allclose(g[n], g_full[n]) for n in g if g[n] is not None), "truncation is real"
    for window in (K, K + 3, 0, None):  # covers the whole loop -> exactly full BPTT
        m = model(feedback=feedback)
        m.mux_loss(q, None, None, trace, ans, n_latents=K, bptt=window)[0].backward()
        g = grads(m)
        assert all((g[n] is None and g_full[n] is None) or torch.equal(g[n], g_full[n]) for n in g), window


def saved_bytes(m, K, window):
    """Bytes autograd keeps for backward during one MUX forward."""
    total = 0

    def pack(t):
        nonlocal total
        total += t.numel() * t.element_size()
        return t

    q, ans, trace = batch(lt=16)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        m.mux_loss(q, None, None, trace, ans, n_latents=K, bptt=window)
    return total


def test_truncation_saves_activation_memory():
    m = model(gamma=0.0)  # continuous path only
    K = 8
    full, w4, w2, w1 = (saved_bytes(m, K, w) for w in (None, 4, 2, 1))
    assert full > w4 > w2 > w1
    # each truncated latent step frees roughly one step's activations
    per_step = (w2 - w1)
    assert abs((w4 - w2) - 2 * per_step) <= 0.05 * (w4 - w2)


def test_truncation_needs_the_cache_path():
    q, ans, _ = batch()
    with pytest.raises(ValueError, match="KV-cache path"):
        model().forward_continuous(q, ans, use_cache=False, n_latents=4, bptt=2)
    model().forward_continuous(q, ans, use_cache=False, n_latents=4, bptt=4)  # no truncation: fine


def test_generate_accepts_any_number_of_loops():
    q, _, _ = batch()
    m = model().eval()
    outs = {k: m.generate(q, 3, n_latents=k) for k in (1, 4, 9)}
    assert all(o.shape == (2, 3) for o in outs.values())


# --------------------------------------------- local loss on p (LOCAL)


def test_local_mode_resolution():
    assert model(feedback="expected").local_mode() == "probs"
    assert model(feedback="projected").local_mode() == "latent"
    assert model(feedback="expected", local_loss="latent").local_mode() == "latent"
    assert model(feedback="projected", local_loss="probs").local_mode() == "probs"
    with pytest.raises(ValueError, match="unknown local_loss"):
        model(local_loss="nope").local_mode()


def test_probs_local_loss_is_kl_to_the_distribution_that_built_the_latent():
    q, ans, trace = batch()
    m = model(feedback="expected", feedback_opts={"tau": 0.7}, gamma=0.0)
    K = 3
    spans, mask, _ = T.chunk_spans(trace, K, m.cfg.max_span, PAD)
    _, parts = m.mux_loss(q, spans, mask, trace, ans, n_latents=K)
    with torch.no_grad():
        latents, _, _, src = m.forward_continuous(q, ans, n_latents=K, return_sources=True)
        # the sources really built the latents
        torch.testing.assert_close(m.to_latent(src), latents)
        t = T.mux_targets(spans, mask, m.cfg)
        logp = torch.log_softmax(m.lm_head(src) / 0.7, -1)
        kl = (t * (t.clamp(min=1e-8).log() - logp)).sum(-1)
    assert parts["local"] == pytest.approx(kl[t.sum(-1) > 0].mean().item(), rel=1e-5)


def test_probs_local_loss_can_reach_zero_but_decoding_the_latent_cannot():
    """With one-token spans and a model that puts all mass on that token,
    KL(target || p) -> 0, while decoding x = p @ E stays flat -- WHEN the
    embedding rows have norm ~1, as in the warm-started codeparrot-small
    (rms 0.036 at d=768). The decoded logit of the target is |E_t|^2, so
    with large embeddings (the std-1 from-scratch init) decoding can be
    sharp: the problem is the warm-start embedding scale."""
    m = model(feedback="expected", gamma=0.0)
    with torch.no_grad():
        m.tok_emb.weight.div_(m.tok_emb.weight.norm(dim=-1, keepdim=True))  # unit rows
        h = torch.randn(1, 1, 32)
        target_tok = int(m.lm_head(h).argmax())
        sharp = h * 200  # p ~ one-hot on target_tok
        t = torch.zeros(1, 1, 97); t[..., target_tok] = 1
        kl = lambda logp: (t * (t.clamp(min=1e-8).log() - logp)).sum(-1).item()
        on_p = kl(torch.log_softmax(m.lm_head(sharp), -1))
        on_x = kl(m.mux_logits(m.to_latent(sharp)))
    assert on_p < 1e-3 < on_x


def test_truncation_still_leaves_values_unchanged_in_probs_mode():
    q, ans, trace = batch()
    a, b = model(feedback="expected"), model(feedback="expected")
    la, pa = a.mux_loss(q, None, None, trace, ans, n_latents=5)
    lb, pb = b.mux_loss(q, None, None, trace, ans, n_latents=5, bptt=1)
    assert torch.equal(la, lb) and pa == pb


# -------------------------------------- global loss + gamma schedule (GLOBAL)


def test_global_distances():
    m_rel, m_cos, m_mse = (model(global_loss=k) for k in ("relative", "cosine", "mse"))
    h_d = torch.randn(4, 32)
    h_c = h_d + 0.1 * torch.randn(4, 32)
    rel = m_rel.global_distance(h_c, h_d)
    manual = ((h_c - h_d).norm(dim=-1) ** 2 / h_d.norm(dim=-1) ** 2).mean()
    torch.testing.assert_close(rel, manual)
    # scale-free: rescaling both states changes MSE, not relative / cosine
    for c in (0.01, 100.0):
        torch.testing.assert_close(m_rel.global_distance(c * h_c, c * h_d), rel)
        torch.testing.assert_close(m_cos.global_distance(c * h_c, c * h_d), m_cos.global_distance(h_c, h_d))
        assert not torch.allclose(m_mse.global_distance(c * h_c, c * h_d), m_mse.global_distance(h_c, h_d))
    # cosine: direction only (each side may scale independently)
    torch.testing.assert_close(m_cos.global_distance(3 * h_c, h_d), m_cos.global_distance(h_c, h_d))
    assert m_rel.global_distance(h_d, h_d) == 0 and m_cos.global_distance(h_d, h_d).abs() < 1e-6
    with pytest.raises(ValueError, match="unknown global_loss"):
        model(global_loss="nope").global_distance(h_c, h_d)


def test_gamma_schedule_shape():
    m = model(gamma=10.0, gamma_warmup=3, gamma_ramp=4)
    seen = []
    for calls in range(10):
        m.train_calls = calls
        seen.append(m.gamma_now())
    assert seen == [0, 0, 0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0]
    step = model(gamma=10.0, gamma_warmup=2)
    step.train_calls = 1
    assert step.gamma_now() == 0
    step.train_calls = 2
    assert step.gamma_now() == 10.0
    assert model(gamma=7.0).gamma_now() == 7.0, "no schedule = the configured gamma"


def test_gamma_counter_counts_training_calls_only():
    q, ans, trace = batch()
    m = model(gamma=1.0, gamma_warmup=2)
    m.eval()
    m.mux_loss(q, None, None, trace, ans)
    m.train()
    with torch.no_grad():
        m.mux_loss(q, None, None, trace, ans)
    assert m.train_calls == 0, "eval and no_grad calls don't advance the schedule"
    gammas = [m.mux_loss(q, None, None, trace, ans)[1]["gamma"] for _ in range(3)]
    assert gammas == [0.0, 0.0, 1.0] and m.train_calls == 3


def test_warmup_turns_off_the_global_term_but_keeps_ref_training():
    q, ans, trace = batch()
    warm = model(gamma=5.0, gamma_warmup=10, feedback="expected", global_loss="relative")
    loss, parts = warm.mux_loss(q, None, None, trace, ans)
    assert parts["gamma"] == 0.0 and parts["global"] > 0, "global is still measured"
    expected = parts["answer"] + warm.cfg.beta * parts["local"] + warm.cfg.ref_answer_weight * parts["ref"]
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    loss.backward()
    assert warm.blocks[0].qkv.weight.grad.abs().sum() > 0
    # the ref (discrete trace) term still trains while gamma is 0
    only_ref = model(gamma=5.0, gamma_warmup=10, feedback="expected")
    _, _ = only_ref.mux_loss(q, None, None, trace, ans)
    disc_logits, _ = only_ref.forward_discrete(q, trace, ans)
    assert disc_logits.requires_grad
