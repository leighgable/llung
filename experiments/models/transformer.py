"""
    Architecture experiment features:
        - Multiplexed Latent Reasoning (MUX, docs/mux.pdf)
        - (Hierarchical ?) Next Latent prediction
        - Residual stream injected memory
        - Looping
        - Relu / Top-K sparse activation
        - Relative Learned Positional Encoding (Inkling's ``InklingRelativeLogits``)
        - Prefix Sliding Attention (prefix.modeling.integration / Triton kernels)
        - Some fast KV Cache (the latent loop and ``generate`` reuse a KV cache)
        - Lightweight Active multimodal

    Fixes relative to the first sketch are marked ``FIX:``. Summary:
      1. RelBias.forward unpacked ``B, T = hidden.shape`` (hidden is [B, T, D]).
      2. ``nn.init_zeros_`` -> ``nn.init.zeros_`` (warm_start=True crashed).
      3. Prefix sliding attention was configured but never applied.
      4. MUX targets were normalized over ``max_span`` instead of each span's
         own length (Eq. 2), so short spans got targets summing to < 1.
      5. The reference (discrete) loss covered only the answer; MUX/CODI train
         the discrete mode with next-token prediction on the whole trace.
      6. Duplicate ``self.n_heads`` assignment.
    Not bugs, but changed: positions are passed to the relative bias (as
    Inkling does) so it stays correct with a KV cache, and the latent loop uses
    a KV cache instead of re-running the trunk on the whole sequence for every
    latent (identical results and gradients, O(K) less work).

    SWAP: two pluggable seams (config keys on Cfg; defaults reproduce the
    original model exactly):
      - residual / residual_opts: how each sublayer reads from and writes to
        the residual stream (models/residual.py: prenorm, scaled, sandwich,
        postnorm, hyper[-connections / mHC]).
      - feedback / feedback_opts: what a latent step feeds into the base of
        the stream (models/feedback.py: projected, expected).

    LOOP: the latent loop is the model's recurrence (the same trunk applied
    once per latent step). Two training controls (Cfg.loop_*):
      - stochastic depth: the number of latent steps K is sampled per training
        call (``sample_n_latents``: fixed | poisson | lognormal_poisson |
        uniform, clipped to [loop_min, loop_max]); eval uses ``n_latents``,
        and ``generate(n_latents=...)`` runs any K at test time.
      - truncated BPTT: only the last ``loop_bptt`` latent steps keep their
        autograd graph; earlier steps run under no_grad (their activations
        are never stored; their KV-cache entries become constants). Forward
        values are identical; only gradients are truncated.
    With a random K, the MUX spans are re-derived from the trace for each K
    (``chunk_spans``).

    Batching: questions and traces must be equal-length within a batch (or
    use batch size 1). There is no attention padding mask yet, so pad tokens
    *inside* a sequence would be attended to. Answer padding at the end is fine
    (causal attention + ``ignore_index``).
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from feedback import make_feedback  # SWAP: latent feedback designs
from residual import Residual, make_residual  # SWAP: residual stream designs


@dataclass
class Cfg:
    vocab_size: int = 32000
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    n_layers: int = 12
    dropout: float = 0.0
    max_len: int = 1024
    # MUX
    n_latents: int = 6            # K latent tokens (paper: 6)
    max_span: int = 16            # S, max subwords per span
    weighting: str = "geometric"  # geometric | sinusoidal | rotary
    rho: float = 0.9              # geometric decay (paper default)
    lam: float = 1.0              # sinusoidal/rotary scale (paper: 1.0)
    rotary_base: float = 1000.0   # rotary frequencies base (paper: 1000)
    tau: float = 1.0              # decoder temperature (paper: 1.0)
    beta: float = 1.0             # local loss weight  (paper: 1.0)
    gamma: float = 20.0           # global distill weight (paper: 20 for LLaMA)
    ref_answer_weight: float = 1.0
    # Inkling
    d_rel: int = 32               # per-head relative-feature size
    rel_extent: int = 256         # bias support: distances < rel_extent
    # prefix sliding window attn
    prefix_attention_window: int = 1024
    # FIX: the prefix is the prompt (paper), whose length varies per example.
    # None = use the question length at runtime; an int fixes it.
    prefix_length: int | None = None
    attn_impl: str = "auto"       # "triton" (GPU kernels) | "eager" (reference) | "auto"
    # SWAP: residual stream design (models/residual.py) and latent feedback
    # (models/feedback.py). Defaults = the original model.
    residual: str = "prenorm"     # prenorm | scaled | sandwich | postnorm | hyper
    residual_opts: dict = field(default_factory=dict)   # e.g. {n: 4, sinkhorn: true}
    feedback: str = "projected"   # projected | expected
    feedback_opts: dict = field(default_factory=dict)   # e.g. {tau: 1.0, top_k: 8}
    # LOOP: stochastic depth over the latent loop + truncated BPTT. Training
    # samples K latent steps per call; eval / generate default to n_latents.
    loop_sampler: str = "fixed"   # fixed | poisson | lognormal_poisson | uniform
    loop_mean: float = 4.0        # E[K] before clipping (poisson: 1 + Poisson(mean - 1))
    loop_sigma: float = 0.5       # lognormal_poisson: log-normal spread of the Poisson rate
    loop_min: int = 1
    loop_max: int = 8
    loop_bptt: int = 0            # backprop through the last N latent steps only (0 = all)
    # LOCAL: what the MUX local loss compares with each slot's span target.
    #   latent: softmax(W x / tau) -- decode the fed-back latent x (paper Eq. 3)
    #   probs:  p = softmax(lm_head(h) / tau), the distribution that BUILT x
    #           (for feedback "expected": x = p @ E lies in the convex hull of
    #           the embeddings, |E x| <~ 1, so decoding x is ~uniform)
    #   auto:   probs for feedback "expected", latent otherwise
    local_loss: str = "auto"
    # GLOBAL: the distance between the continuous and the (stop-grad) discrete
    # answer state, and a schedule for its weight gamma.
    #   mse:      F.mse_loss (paper; per-dimension, so gamma depends on the
    #             hidden-state scale -- gamma=20 was tuned for LLaMA)
    #   relative: ||h_c - h_d||^2 / ||h_d||^2   (scale-free: "fraction off")
    #   cosine:   1 - cos(h_c, h_d)             (direction only)
    global_loss: str = "mse"
    # gamma is 0 for the first gamma_warmup training calls (the discrete path
    # learns the task first, so the target stops moving), then ramps linearly
    # to `gamma` over gamma_ramp calls. A training call = one mux_loss with
    # grad in train mode (optimizer steps x grad_accum); the counter is not
    # checkpointed, so every run (stage) starts its own schedule.
    gamma_warmup: int = 0
    gamma_ramp: int = 0


class RelBias(nn.Module):
    """Query-conditioned learned relative bias added to attention logits.

    Same as Inkling's ``InklingRelativeLogits`` (transformers): each query
    mixes a trained bank of bias-vs-distance profiles (``bias_bank``, Inkling's
    ``proj``) into one bias per backward distance; zero outside
    ``0 <= distance < rel_extent``. Causality/masking stays in the mask.
    """

    def __init__(self, cfg: Cfg, zero_init: bool = False):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.rel_extent = cfg.rel_extent
        self.r_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_rel, bias=False)
        self.bias_bank = nn.Parameter(torch.randn(cfg.d_rel, cfg.rel_extent) * 0.02)
        if zero_init:
            nn.init.zeros_(self.r_proj.weight)  # FIX: was nn.init_zeros_

    def profile(self, hidden: torch.Tensor) -> torch.Tensor:
        """Per-query profile ``[B, H, T, rel_extent]`` (what the kernels consume)."""
        B, T, _ = hidden.shape
        rel = self.r_proj(hidden).view(B, T, self.n_heads, -1)
        return torch.einsum("bthd,de->bhte", rel, self.bias_bank)

    def forward(self, hidden: torch.Tensor, q_positions=None, k_positions=None) -> torch.Tensor:
        """Dense bias ``[B, H, Tq, Tk]`` for the eager path.

        Positions are absolute (Inkling passes cache offsets the same way);
        default: ``0..T-1`` for both, i.e. a full sequence without a cache.
        """
        B, T, _ = hidden.shape  # FIX: was `B, T = hidden.shape`
        profile = self.profile(hidden)
        if q_positions is None:
            q_positions = torch.arange(T, device=hidden.device)
        if k_positions is None:
            k_positions = q_positions
        dist = q_positions[:, None] - k_positions[None, :]
        idx = dist.clamp(0, self.rel_extent - 1).expand(B, self.n_heads, -1, -1)
        bias = profile.gather(-1, idx)
        outside = (dist < 0) | (dist >= self.rel_extent)
        return bias.masked_fill(outside[None, None], 0.0)


def eager_prefix_sliding_attention(q, k, v, bias, q_positions, k_positions, prefix_length: int, window: int):
    """Reference attention: softmax(q k^T / sqrt(d) + bias) under the
    prefix-sliding mask (k <= q and (k < P or k > q - W)). Explicit math, no
    SDPA: fp32 SDPA with a float mask is wrong on ROCm."""
    qp, kp = q_positions[:, None], k_positions[None, :]
    allowed = (kp <= qp) & ((kp < prefix_length) | (kp > qp - window))
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + bias
    scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(-1) @ v


class Block(nn.Module):
    def __init__(self, cfg: Cfg, zero_init_rel: bool, residual: Residual | None = None, index: int = 0):
        super().__init__()
        self.n_heads = cfg.n_heads  # FIX: was assigned twice
        self.head_dim = cfg.d_model // cfg.n_heads
        self.window = cfg.prefix_attention_window
        self.attn_impl = cfg.attn_impl
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rel_bias = RelBias(cfg, zero_init=zero_init_rel)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model),
        )
        # SWAP: how the two sublayers read from / write to the stream. The
        # LayerNorms stay here (ln1 / ln2 keys), so checkpoints load under
        # every design. prenorm has no parameters.
        residual = residual if residual is not None else make_residual("prenorm")
        self.res_attn = residual.sublayer(2 * index, cfg.d_model)
        self.res_ffn = residual.sublayer(2 * index + 1, cfg.d_model)

    def use_triton(self, x: torch.Tensor) -> bool:
        return self.attn_impl == "triton" or (self.attn_impl == "auto" and x.is_cuda)

    def attention(self, h: torch.Tensor, prefix_length: int, cache=None, layer_idx=None) -> torch.Tensor:
        """Prefix sliding attention with the learned relative bias -> [B, T, D].

        FIX: the sketch applied a plain causal mask; prefix sliding was never used.
        ``cache``: any transformers ``Cache`` (``DynamicCache`` for training,
        ``PrefixSlidingRingCache`` for fast decoding); ``h`` then holds only
        the new tokens and positions continue from the cache.
        """
        if self.use_triton(h):
            from prefix.modeling.integration import block_attention
            from prefix.modeling.monkeypatch import PrefixSlidingConfig

            ps = PrefixSlidingConfig(prefix_length, self.window)
            return block_attention(self, h, ps, cache=cache, layer_idx=layer_idx)

        B, T, D = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        offset = 0
        if cache is not None:
            offset = cache.get_seq_length(layer_idx)
            k, v = cache.update(k, v, layer_idx)
            if k.shape[2] != offset + T:
                raise NotImplementedError("the eager path needs a cache that keeps the full history (DynamicCache)")
        q_pos = torch.arange(T, device=h.device) + offset
        k_pos = torch.arange(k.shape[2], device=h.device)
        bias = self.rel_bias(h, q_pos, k_pos)
        a = eager_prefix_sliding_attention(q, k, v, bias, q_pos, k_pos, prefix_length, self.window)
        return a.transpose(1, 2).reshape(B, T, D)

    def forward(self, x, prefix_length: int, cache=None, layer_idx=None):
        """``x`` is the residual stream state (its shape is the residual
        design's business; [B, T, D] for single-stream designs).
        SWAP: was ``x = x + o_proj(attn(ln1(x)))`` / ``x = x + ff(ln2(x))``,
        which is exactly what the default ``prenorm`` design computes."""
        x = self.res_attn(x, lambda h: self.o_proj(self.attention(h, prefix_length, cache, layer_idx)), self.ln1)
        x = self.res_ffn(x, self.ff, self.ln2)
        return x


def positional_weights(kind: str, S: int, rho: float, lam: float,
                       rotary_base: float, n_freqs: int = 8) -> torch.Tensor:
    """Normalized weights alpha_j = w_j / sum(w), j = 1..S, for ONE span of length S."""
    j = torch.arange(S, dtype=torch.float32)
    if kind == "geometric":
        w = rho ** j                                   # w_j = rho^{j-1}
    elif kind == "sinusoidal":
        s = torch.sin(math.pi / 2 * j / max(S - 1, 1)) # peaks at span end
        w = torch.exp(lam * s)
    elif kind == "rotary":
        # Paper (Cor. 6): lossless if 0 < theta_p * (S - 1) < pi for all p.
        theta = rotary_base ** (-2 * torch.arange(n_freqs).float() / n_freqs)
        s = torch.cos(j[:, None] * theta[None, :]).mean(-1)
        w = torch.exp(lam * s)
    else:
        raise ValueError(kind)
    return w / w.sum()


def positional_weight_table(cfg: Cfg) -> torch.Tensor:
    """``[S+1, S]``: row L = normalized weights of a span of length L (zero-padded).

    FIX: Eq. 2 normalizes each span by its OWN length, alpha_j = w_j / sum_{l<=S_i} w_l
    (and the sinusoidal scores depend on S_i too). The sketch normalized every
    span over ``max_span``, so short spans' targets summed to < 1.
    """
    S = cfg.max_span
    table = torch.zeros(S + 1, S)
    for L in range(1, S + 1):
        table[L, :L] = positional_weights(cfg.weighting, L, cfg.rho, cfg.lam, cfg.rotary_base)
    return table


def mux_targets(span_ids: torch.Tensor, span_mask: torch.Tensor,
                cfg: Cfg) -> torch.Tensor:
    """Build superposition targets for every (batch, latent-slot).

    span_ids: [B, K, S] token ids of each slot's aligned span (padded)
    span_mask: [B, K, S] True for real subwords; must be a prefix of each row
        (span tokens first, padding after)
    -> targets: [B, K, V] in the vocabulary simplex (rows sum to 1, or are 0
       for slots without an aligned span).
    """
    B, K, S = span_ids.shape
    lengths = span_mask.sum(-1)                                  # S_i per slot
    alpha = positional_weight_table(cfg).to(span_ids.device)[lengths]  # [B, K, S], per-span normalized
    w = alpha * span_mask.float()
    t = torch.zeros(B, K, cfg.vocab_size, device=span_ids.device)
    t.scatter_add_(2, span_ids.clamp(min=0), w)
    return t  # rows of empty slots are already all-zero


def chunk_spans(trace_ids: torch.Tensor, K: int, max_span: int, pad_id: int):
    """LOOP: split each trace into K contiguous, near-equal spans (for a
    sampled number of latent steps). Token p of a trace with L real tokens
    goes to slot floor(p * K / L); spans longer than ``max_span`` keep their
    first ``max_span`` tokens. Traces must be pad-terminated (pads at the end).

    Returns span_ids [B, K, S] (pad-filled), span_mask [B, K, S] (a prefix of
    each row, as mux_targets requires) and the fraction of trace tokens that
    fit (< 1 when K * max_span < L).
    """
    B, T = trace_ids.shape
    device = trace_ids.device
    L = (trace_ids != pad_id).sum(1)                                   # [B]
    pos = torch.arange(T, device=device).expand(B, T)
    slot = (pos * K) // L.clamp(min=1)[:, None]                        # [B, T]
    start = (slot * L[:, None] + K - 1) // K                           # ceil(slot * L / K)
    idx = pos - start
    valid = (pos < L[:, None]) & (idx < max_span) & (slot < K)
    span_ids = torch.full((B, K, max_span), pad_id, dtype=trace_ids.dtype, device=device)
    span_mask = torch.zeros(B, K, max_span, dtype=torch.bool, device=device)
    b = torch.arange(B, device=device)[:, None].expand(B, T)
    span_ids[b[valid], slot[valid], idx[valid]] = trace_ids[valid]
    span_mask[b[valid], slot[valid], idx[valid]] = True
    coverage = valid.sum().float() / L.sum().clamp(min=1)
    return span_ids, span_mask, coverage.item()


def sample_loops(cfg: "Cfg", generator: torch.Generator | None = None) -> int:
    """LOOP: one draw of K from cfg.loop_sampler, clipped to [loop_min, loop_max].

    poisson:            K = 1 + Poisson(loop_mean - 1)
    lognormal_poisson:  K = 1 + Poisson(exp(tau)), tau ~ N(log(loop_mean - 1) - sigma^2/2, sigma)
                        (heavy-tailed; the recurrent-depth / retrofitted-recurrence recipe,
                        centred so that E[K] = loop_mean like the other samplers)
    uniform:            K ~ U{loop_min, ..., loop_max}
    fixed:              K = n_latents
    """
    kind, lo, hi = cfg.loop_sampler, int(cfg.loop_min), int(cfg.loop_max)
    if kind == "fixed":
        return int(cfg.n_latents)
    if kind == "poisson":
        k = 1 + torch.poisson(torch.tensor(max(cfg.loop_mean - 1.0, 0.0)), generator=generator).item()
    elif kind == "lognormal_poisson":
        sigma = float(cfg.loop_sigma)
        rate = max(cfg.loop_mean - 1.0, 1e-6)  # E[exp(tau)] = rate -> E[K] = loop_mean
        tau = math.log(rate) - 0.5 * sigma**2 + sigma * torch.randn((), generator=generator).item()
        k = 1 + torch.poisson(torch.tensor(math.exp(tau)), generator=generator).item()
    elif kind == "uniform":
        k = torch.randint(lo, hi + 1, (), generator=generator).item()
    else:
        raise ValueError(f"unknown loop_sampler '{kind}' (fixed | poisson | lognormal_poisson | uniform)")
    return int(min(max(k, lo), hi))


class LatentLM(nn.Module):
    def __init__(self,
        cfg: Cfg, warm_start: bool = False,
        bot_id: int = 1, eot_id: int = 2, pad_id: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.bot_id, self.eot_id, self.pad_id = bot_id, eot_id, pad_id
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.residual = make_residual(cfg.residual, **(cfg.residual_opts or {}))  # SWAP
        self.blocks = nn.ModuleList(
            Block(cfg, zero_init_rel=warm_start, residual=self.residual, index=i) for i in range(cfg.n_layers)
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # SWAP: was latent_proj + std_norm (= feedback "projected", scale
        # "unit_std"; the weight moved to feedback.proj.weight).
        self.feedback = make_feedback(cfg.feedback, cfg.d_model, **(cfg.feedback_opts or {}))
        self.train_calls = 0  # GLOBAL: drives the gamma schedule (see Cfg)

    def prefix_length(self, q_ids: torch.Tensor) -> int:
        """The always-visible prefix: ``cfg.prefix_length`` or the question."""
        return self.cfg.prefix_length if self.cfg.prefix_length is not None else q_ids.shape[1]

    def trunk(self, embeds: torch.Tensor, prefix_length: int, cache=None) -> torch.Tensor:
        """``embeds`` = the new positions only when a ``cache`` is given."""
        h = self.residual.begin(embeds)  # SWAP: the design shapes the stream
        for i, blk in enumerate(self.blocks):
            h = blk(h, prefix_length, cache=cache, layer_idx=i)
        return self.final_ln(self.residual.end(h))

    def to_latent(self, h_last: torch.Tensor) -> torch.Tensor:
        """ hidden fed back as next input embedding (SWAP: models/feedback.py) """
        return self.feedback(h_last, self)

    def sample_n_latents(self) -> int:
        """LOOP: K for this call -- sampled in training mode, n_latents in eval."""
        return sample_loops(self.cfg) if self.training else int(self.cfg.n_latents)

    def local_mode(self) -> str:
        mode = self.cfg.local_loss
        if mode == "auto":
            return "probs" if self.cfg.feedback == "expected" else "latent"
        if mode not in ("latent", "probs"):
            raise ValueError(f"unknown local_loss '{mode}' (auto | latent | probs)")
        return mode

    def local_logp(self, latents: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        """LOCAL: log-distribution the MUX local loss compares with the span
        targets. probs: log softmax(lm_head(h) / tau) with the feedback's own
        tau (full vocab: a top-k cut would put -inf under target mass)."""
        if self.local_mode() == "latent":
            return self.mux_logits(latents)
        tau = getattr(self.feedback, "tau", None) or self.cfg.tau
        return F.log_softmax(self.lm_head(sources).float() / tau, dim=-1)

    def gamma_now(self) -> float:
        """GLOBAL: the scheduled weight of the global loss for this call."""
        c = self.cfg
        done = self.train_calls - int(c.gamma_warmup)
        if done < 0:
            return 0.0
        if int(c.gamma_ramp) <= 0:
            return float(c.gamma)
        return float(c.gamma) * min(1.0, (done + 1) / int(c.gamma_ramp))

    def global_distance(self, h_cont: torch.Tensor, h_disc: torch.Tensor) -> torch.Tensor:
        """GLOBAL: distance between answer states (h_disc already detached)."""
        kind = self.cfg.global_loss
        if kind == "mse":
            return F.mse_loss(h_cont, h_disc)
        h_cont, h_disc = h_cont.float(), h_disc.float()
        if kind == "relative":
            return ((h_cont - h_disc).pow(2).sum(-1) / h_disc.pow(2).sum(-1).clamp(min=1e-8)).mean()
        if kind == "cosine":
            return (1 - F.cosine_similarity(h_cont, h_disc, dim=-1)).mean()
        raise ValueError(f"unknown global_loss '{kind}' (mse | relative | cosine)")

    def mux_logits(self, latents: torch.Tensor) -> torch.Tensor:
        """f(x) = softmax(W x / tau) using the tied unembedding (Eq. 3)."""
        return F.log_softmax(self.lm_head(latents) / self.cfg.tau, dim=-1)

    def _eot(self, q_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(torch.full((q_ids.shape[0], 1), self.eot_id, dtype=torch.long, device=q_ids.device))

    # teacher forcing
    def forward_continuous(self, q_ids: torch.Tensor, answer_ids: torch.Tensor, use_cache: bool = True,
                           n_latents: int | None = None, bptt: int | None = None, return_sources: bool = False):
        """Run the latent loop. q_ids: [B, Lq] should END with <bot>.
        answer_ids: [B, La] discrete answer tokens.
        Returns latents [B, K, D], answer_logits [B, La, V], h_answer [B, D].

        LOOP: ``n_latents`` = K for this call (default cfg.n_latents).
        ``bptt`` = N: only the last N latent steps keep their autograd graph
        (truncated BPTT; None / 0 / >= K: full backprop). Steps before the
        window run under no_grad: their activations are never stored and their
        cache entries are constants to the steps after. The question prefill
        and the answer pass always keep gradients. Needs the KV-cache path.

        LOCAL: ``return_sources=True`` also returns the hidden states that
        built each latent ([B, K, D], what ``to_latent`` was applied to).

        use_cache=True (default) runs each latent step on the new position only,
        reusing keys/values from a ``DynamicCache``. It keeps the full history,
        so gradients flow as before, and prefix sliding masks the middle.
        Identical results to use_cache=False (re-running the trunk on the whole
        sequence for every latent), at a fraction of the cost.
        """
        K = int(n_latents if n_latents is not None else self.cfg.n_latents)
        if K < 1:
            raise ValueError(f"n_latents must be >= 1, got {K}")
        first_grad = max(0, K - int(bptt)) if bptt else 0   # LOOP: steps < first_grad: no graph
        if first_grad and not use_cache:
            raise ValueError("truncated BPTT is defined on the KV-cache path (use_cache=True)")
        P = self.prefix_length(q_ids)
        q_emb = self.tok_emb(q_ids)
        tail = torch.cat([self._eot(q_ids), self.tok_emb(answer_ids)[:, :-1]], dim=1)
        latents, sources = [], []
        if use_cache:
            from transformers import DynamicCache

            cache = DynamicCache()
            h = self.trunk(q_emb, P, cache)
            grad_on = torch.is_grad_enabled()
            for k in range(K):                             # the loop: same trunk,
                with torch.set_grad_enabled(grad_on and k >= first_grad):  # LOOP: truncation
                    sources.append(h[:, -1])
                    x = self.to_latent(h[:, -1])               # new latent token
                    latents.append(x)
                    if k < K - 1:
                        h = self.trunk(x.unsqueeze(1), P, cache)   # feed back (1 new position)
            # answer pass: last latent, <eot>, then answer tokens (teacher forcing)
            h = self.trunk(torch.cat([x.unsqueeze(1), tail], dim=1), P, cache)
        else:
            embeds = q_emb
            for _ in range(K):                             # the loop: same trunk,
                h = self.trunk(embeds, P)                  # recurrently applied
                sources.append(h[:, -1])
                x = self.to_latent(h[:, -1])               # new latent token
                latents.append(x)
                embeds = torch.cat([embeds, x.unsqueeze(1)], dim=1)   # feed back
            h = self.trunk(torch.cat([embeds, tail], dim=1), P)
        h_ans = h[:, -answer_ids.shape[1]:]            # positions predicting answer
        answer_logits = self.lm_head(h_ans)
        if return_sources:
            return torch.stack(latents, dim=1), answer_logits, h_ans[:, 0], torch.stack(sources, dim=1)
        return torch.stack(latents, dim=1), answer_logits, h_ans[:, 0]

    def forward_discrete(self, q_ids: torch.Tensor, trace_ids: torch.Tensor,
                         answer_ids: torch.Tensor):
        """
            Discrete-reasoning pass, used for global distillation loss and the
            reference loss.

            Returns (logits [B, Lt + 1 + La, V] predicting trace, <eot>, answer;
            the hidden state predicting the first answer token [B, D]).
        """
        eot = torch.full((q_ids.shape[0], 1), self.eot_id, dtype=torch.long,
                         device=q_ids.device)
        ids = torch.cat([q_ids, trace_ids, eot, answer_ids[:, :-1]], dim=1)
        h = self.trunk(self.tok_emb(ids), self.prefix_length(q_ids))
        # FIX: return predictions for the whole trace too (from the last question token on)
        logits = self.lm_head(h[:, q_ids.shape[1] - 1:])
        return logits, h[:, -answer_ids.shape[1]]

    # ---- composite MUX loss ---------------------------------------------- #
    def mux_loss(self, q_ids, span_ids, span_mask, trace_ids, answer_ids,
                 n_latents: int | None = None, bptt: int | None = None):
        """LOOP: K = n_latents, else span_ids.shape[1], else cfg.n_latents.
        When span_ids is None or has a different K, spans are re-derived from
        the trace (chunk_spans) and parts["span_coverage"] reports how much of
        the trace fits. ``bptt``: see forward_continuous. Loss VALUES do not
        depend on bptt (every latent slot is in the local loss); the gradient
        comes from the last ``bptt`` slots only.
        """
        K = n_latents if n_latents is not None else (span_ids.shape[1] if span_ids is not None else self.cfg.n_latents)
        extra = {}
        if span_ids is None or span_ids.shape[1] != K:
            span_ids, span_mask, extra["span_coverage"] = chunk_spans(trace_ids, K, self.cfg.max_span, self.pad_id)
        latents, ans_logits, h_ans_cont, sources = self.forward_continuous(
            q_ids, answer_ids, n_latents=K, bptt=bptt, return_sources=True)
        targets = mux_targets(span_ids, span_mask, self.cfg)          # [B,K,V]
        logp = self.local_logp(latents, sources)                      # [B,K,V]
        nonempty = targets.sum(-1) > 0                                # [B,K]
        kl = -(targets * logp).sum(-1) + (targets * targets.clamp(min=1e-8).log()).sum(-1)
        l_local = kl[nonempty].mean() if nonempty.any() else latents.sum() * 0.0
        l_answer = F.cross_entropy(ans_logits.reshape(-1, ans_logits.size(-1)),
                                   answer_ids.reshape(-1),
                                   ignore_index=self.pad_id)
        loss = l_answer + self.cfg.beta * l_local
        gamma = self.gamma_now()  # GLOBAL: scheduled weight
        if self.training and torch.is_grad_enabled():
            self.train_calls += 1
        l_global = torch.tensor(0.0, device=q_ids.device)
        l_ref = torch.tensor(0.0, device=q_ids.device)
        ref_extra = {}
        if self.cfg.gamma > 0:
            ref_logits, h_ans_disc = self.forward_discrete(q_ids, trace_ids, answer_ids)
            # Paper: ||h_cont - sg(h_disc)||^2; mse_loss also averages over d_model
            # (a constant 1/d rescaling of gamma). GLOBAL: cfg.global_loss picks
            # the distance; the ref (discrete) loss is on whenever cfg.gamma > 0,
            # including during gamma's warmup.
            l_global = self.global_distance(h_ans_cont, h_ans_disc.detach())    # stop-grad
            # FIX: the discrete mode is trained with next-token prediction on the
            # whole discrete trace (MUX Sec. 3.3 / App. B.4, following CODI), not
            # only on the answer: targets = trace, <eot>, answer.
            eot = torch.full((q_ids.shape[0], 1), self.eot_id, dtype=torch.long, device=q_ids.device)
            ref_targets = torch.cat([trace_ids, eot, answer_ids], dim=1)
            l_ref = F.cross_entropy(ref_logits.reshape(-1, ref_logits.size(-1)),
                                    ref_targets.reshape(-1), ignore_index=self.pad_id)
            # GLOBAL: can the discrete teacher answer? First answer token, TEACHER-
            # FORCED on the written trace -- when the answer is restated in the
            # trace (program_trace: the last value) this measures copying, not
            # reasoning; check the trace tokens themselves for that.
            with torch.no_grad():
                La = answer_ids.shape[1]
                ref_ok = ref_logits[:, -La].argmax(-1) == answer_ids[:, 0]
                ref_extra["ref_answer_acc"] = ref_ok.float().mean().item()
            loss = loss + gamma * l_global + self.cfg.ref_answer_weight * l_ref
        with torch.no_grad():  # LOOP: first-answer-token accuracy (monitoring)
            first_ok = ans_logits[:, 0].argmax(-1) == answer_ids[:, 0]
        return loss, {"answer": l_answer.item(), "local": l_local.item(),
                      "global": l_global.item(), "ref": l_ref.item(),
                      "answer_acc": first_ok.float().mean().item(), "gamma": gamma, **ref_extra, **extra}

    # ---- inference --------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, q_ids: torch.Tensor, max_new_tokens: int, cache=None, n_latents: int | None = None) -> torch.Tensor:
        """Greedy: K latent steps, <eot>, then answer tokens, all incremental.
        LOOP: ``n_latents`` = K (default cfg.n_latents) -- with stochastic-depth
        training, more steps at test time can buy accuracy.

        Default cache: ``PrefixSlidingRingCache`` on the Triton path (fixed-size,
        decode kernel; prefix = the question), ``DynamicCache`` on the eager path.
        Returns answer token ids [B, max_new_tokens].
        """
        P = self.prefix_length(q_ids)
        if cache is None:
            if self.blocks[0].use_triton(q_ids):
                from prefix.modeling.cache import PrefixSlidingRingCache
                from prefix.modeling.monkeypatch import PrefixSlidingConfig

                cache = PrefixSlidingRingCache(PrefixSlidingConfig(P, self.cfg.prefix_attention_window))
            else:
                from transformers import DynamicCache

                cache = DynamicCache()
        h = self.trunk(self.tok_emb(q_ids), P, cache)
        for _ in range(int(n_latents if n_latents is not None else self.cfg.n_latents)):
            h = self.trunk(self.to_latent(h[:, -1]).unsqueeze(1), P, cache)
        h = self.trunk(self._eot(q_ids), P, cache)
        out = []
        for _ in range(max_new_tokens):
            tok = self.lm_head(h[:, -1]).argmax(-1, keepdim=True)
            out.append(tok)
            h = self.trunk(self.tok_emb(tok), P, cache)
        return torch.cat(out, dim=1)

    # ---- warm start ------------------------------------------------------- #
    @torch.no_grad()
    def warm_start_from(self, backbone_sd: dict):
        """Copy matching backbone weights; keep r_proj zero-init so the
        bias starts at exactly 0 and attention matches the backbone at step 0.
        Call with warm_start=True in the constructor."""
        own = self.state_dict()
        copied = skipped = 0
        for k, v in backbone_sd.items():
            if "rel_bias" in k:                    # never import positional bias
                skipped += 1
                continue
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                copied += 1
            else:
                skipped += 1
        print(f"warm-start: copied {copied} tensors, skipped {skipped}")
