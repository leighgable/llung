"""LatentLM's swappable seams: residual stream designs (residual.py)
and latent feedback (feedback.py). Tiny CPU models, eager attention."""

import sys
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings, strategies as st

# No conftest.py here on purpose: core/tests imports its own `conftest`
# by name, and two conftest modules would shadow each other.
MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS))          # transformer, residual, feedback
sys.path.insert(0, str(MODELS.parent))   # codetok

import transformer as T  # noqa: E402
from residual import sinkhorn  # noqa: E402

TINY = dict(vocab_size=97, d_model=32, n_heads=4, d_ff=64, n_layers=3, n_latents=3,
            max_span=4, d_rel=8, rel_extent=16, attn_impl="eager", gamma=1.0)

# warm-start safe: identical to prenorm at init (residual.py docstrings)
SAFE = [
    ("scaled", {}),
    ("hyper", {"n": 2}),
    ("hyper", {"n": 4, "sinkhorn": True}),
    ("hyper", {"n": 3, "dynamic": True}),
    ("hyper", {"n": 4, "sinkhorn": True, "dynamic": True}),
]
ALL = [("prenorm", {}), *SAFE, ("scaled", {"init": 0.0}), ("postnorm", {}), ("sandwich", {})]


def model(residual="prenorm", residual_opts=None, seed=0, **kw):
    torch.manual_seed(seed)
    return T.LatentLM(T.Cfg(**{**TINY, **kw}, residual=residual, residual_opts=residual_opts or {}), warm_start=True)


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(3, 97, (2, 5), generator=g)
    ans = torch.randint(3, 97, (2, 4), generator=g)
    trace = torch.randint(3, 97, (2, 6), generator=g)
    spans = torch.randint(3, 97, (2, 3, 4), generator=g)
    mask = torch.arange(4)[None, None] < torch.tensor([[1, 4, 2], [3, 0, 4]])[..., None]
    return q, ans, trace, spans, mask


def ids(name, opts):
    return f"{name}{opts}"


@pytest.mark.parametrize("name, opts", SAFE, ids=[ids(*c) for c in SAFE])
def test_safe_designs_equal_prenorm_at_init(name, opts):
    """Same core weights -> same function: a warm-started backbone keeps
    its behaviour under every warm-start-safe design."""
    base = model()
    other = model(name, opts)
    missing, unexpected = other.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(".res_" in k for k in missing), missing
    q, ans, trace, *_ = batch()
    with torch.no_grad():
        torch.testing.assert_close(other.trunk(other.tok_emb(q), 5), base.trunk(base.tok_emb(q), 5), atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(other.forward_discrete(q, trace, ans)[0], base.forward_discrete(q, trace, ans)[0],
                                   atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("name, opts", ALL, ids=[ids(*c) for c in ALL])
def test_kv_cache_matches_full_recompute_and_grads_flow(name, opts):
    """The latent loop's KV-cache path is identical to re-running the
    trunk on the whole sequence, for every design -- forward AND backward."""
    q, ans, trace, spans, mask = batch()
    m = model(name, opts)
    for p in m.parameters():  # perturb so designs differ from their init
        p.data.add_(0.05 * torch.randn_like(p))
    with_cache = m.forward_continuous(q, ans, use_cache=True)
    without = m.forward_continuous(q, ans, use_cache=False)
    for a, b in zip(with_cache, without):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    loss, parts = m.mux_loss(q, spans, mask, trace, ans)
    loss.backward()
    assert torch.isfinite(loss)
    res_params = [(n, p) for n, p in m.named_parameters() if ".res_" in n and p.requires_grad]
    for n, p in res_params:
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_hyper_streams_diverge_with_training():
    """The streams start as identical copies; the per-layer read weights
    break the symmetry, so training makes them differ (otherwise n
    streams would just be one stream n times)."""
    for opts in ({"n": 3}, {"n": 3, "sinkhorn": True}):
        m = model("hyper", opts)
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        q, ans, trace, spans, mask = batch()
        for _ in range(3):
            opt.zero_grad()
            m.mux_loss(q, spans, mask, trace, ans)[0].backward()
            opt.step()
        state = m.residual.begin(m.tok_emb(q))
        state = m.blocks[0](state, 5)
        assert (state[..., 0, :] - state[..., 1, :]).abs().max() > 1e-6, opts


@given(st.integers(2, 6), st.integers(0, 10_000), st.floats(0.0, 3.0))
@settings(max_examples=40, deadline=None, derandomize=True)
def test_sinkhorn(n, seed, scale):
    """Columns sum to 1 exactly (the pre-LN-equivalence invariant), all
    entries are non-negative, and more iterations never make the row
    sums worse."""
    logits = 4 * torch.eye(n) + scale * torch.randn(n, n, generator=torch.Generator().manual_seed(seed))
    p = sinkhorn(logits)
    assert (p >= 0).all()
    torch.testing.assert_close(p.sum(0), torch.ones(n), atol=1e-5, rtol=0)
    row_err = lambda iters: (sinkhorn(logits, iters).sum(1) - 1).abs().max().item()
    assert row_err(200) <= row_err(20) + 1e-6 <= row_err(5) + 2e-6
    if scale <= 1.0:
        assert row_err(200) < 1e-2


def test_unknown_names_are_clear_errors():
    with pytest.raises(ValueError, match="unknown residual"):
        model("nope")
    with pytest.raises(ValueError, match="unknown feedback"):
        T.LatentLM(T.Cfg(**TINY, feedback="nope"))


# ------------------------------------------------------------- feedback


def fmodel(feedback, **opts):
    torch.manual_seed(0)
    return T.LatentLM(T.Cfg(**TINY, feedback=feedback, feedback_opts=opts))


def test_expected_feedback_is_the_probability_weighted_embedding():
    m = fmodel("expected")
    h = torch.randn(2, 32)
    p = torch.softmax(m.lm_head(h) / m.cfg.tau, -1)
    torch.testing.assert_close(m.to_latent(h), p @ m.tok_emb.weight)
    assert not list(m.feedback.parameters()), "no parameters of its own"


def test_expected_feedback_top_k_and_tau():
    m = fmodel("expected", top_k=3, tau=0.5)
    h = torch.randn(4, 32)
    p = m.feedback.probs(h, m)
    assert ((p > 0).sum(-1) == 3).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(4))


def test_feedback_gradients_reach_the_trunk_through_the_latent_loop():
    for name in ("expected", "projected"):
        m = fmodel(name)
        q, ans, trace, spans, mask = batch()
        m.mux_loss(q, spans, mask, trace, ans)[0].backward()
        assert m.blocks[0].qkv.weight.grad.abs().sum() > 0, name


def test_projected_feedback_scales():
    h = torch.randn(5, 32)
    orig = fmodel("projected")  # the original LatentLM behaviour
    x = orig.to_latent(h)
    torch.testing.assert_close(x.std(-1), torch.ones(5))
    emb = fmodel("projected", scale="embedding")
    x = emb.to_latent(h)
    target = emb.tok_emb.weight.pow(2).mean().sqrt()
    torch.testing.assert_close(x.pow(2).mean(-1).sqrt(), target.expand(5))
