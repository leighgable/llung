"""LatentLM (transformer.py) as a plain causal LM for the harness.

The engine calls `model(x)`; LatentLM only has the MUX entry points
(forward_continuous / forward_discrete / mux_loss). PretrainLM adds the
discrete next-token forward: ids (B, T) -> logits (B, T, vocab), i.e.
token embeddings -> trunk -> tied LM head, no latent loop.

It SUBCLASSES LatentLM, so parameter names are identical: a checkpoint
from a pretraining run loads straight into LatentLM for MUX training
(`LatentLM(cfg).load_state_dict(torch.load(".../latest.pt")["model"])`).

Config: every key of the model section except `factory` and the
constructor flags below is a field of transformer.Cfg:

    model:
      factory: models/latent_lm.py:PretrainLM
      warm_start: true      # zero-init the relative bias projection
      attn_impl: triton     # prefix kernels (flake input prefix-src); "eager"
                            # = the reference math; "auto" = triton on GPU

The tokenizer is codetok (experiments/codetok): vocab_size and the
special ids (bot/eot/pad) default to it. LatentLM's own defaults
(bot 1, eot 2, pad 0) are "!", '"' and <|endoftext|> in this vocab --
never use them with codetok data.
"""

import codetok
import torch
import torch.nn.functional as F

from transformer import Cfg, LatentLM


class PretrainLM(LatentLM):
    def __init__(
        self,
        warm_start: bool = False,
        bot_id: int = codetok.BOT,
        eot_id: int = codetok.EOT,
        pad_id: int = codetok.PAD,
        vocab_size: int = codetok.VOCAB_SIZE,
        **cfg,
    ):
        if vocab_size < codetok.VOCAB_SIZE:
            raise ValueError(f"vocab_size {vocab_size} < codetok's {codetok.VOCAB_SIZE}")
        super().__init__(
            Cfg(vocab_size=vocab_size, **cfg), warm_start=warm_start, bot_id=bot_id, eot_id=eot_id, pad_id=pad_id
        )

    def forward(self, ids):
        # The prefix-sliding mask with prefix 0 is plain sliding-window
        # causal attention (= full causal while seq_len <= the window).
        prefix = self.cfg.prefix_length or 0
        return self.lm_head(self.trunk(self.tok_emb(ids), prefix))


class MuxLM(PretrainLM):
    """LatentLM trained on its MUX objective through the harness: the
    engine's `training_loss` hook (core/train.py) calls this instead of
    task.loss(model(x)).

    Batches are dicts {q, trace, answer} (+ optional spans / span_mask;
    without them the trace is chunked into K spans per call). The loop
    controls are Cfg fields in the model section:

        model:
          factory: models/latent_lm.py:MuxLM
          n_latents: 4            # K at eval
          loop_sampler: poisson   # stochastic depth (training only)
          loop_mean: 4
          loop_min: 1
          loop_max: 8
          loop_bptt: 2            # truncated BPTT window (0 = full)
          eval_loops: [1, 2, 4, 8]  # answer accuracy at each K (val only)
          objective: mux          # mux | discrete (stage 1: learn the written-out
                                  # trace first -- next-token on trace + answer only)

    parts -> metrics.csv: answer / local / global / ref (loss terms),
    answer_acc (first answer token), loops (K; its epoch mean in
    train_loops), span_coverage, acc_k<K> for each eval_loops entry, and
    trace_acc: the discrete path's teacher-forced accuracy on the trace
    tokens that need computing (x["trace_mask"] if the task provides it,
    else every non-pad trace token). trace_acc is the honest "can the
    teacher reason" number -- ref_answer_acc can be pure copying.
    """

    def __init__(self, eval_loops: list[int] | None = None, objective: str = "mux", **kwargs):
        super().__init__(**kwargs)
        if objective not in ("mux", "discrete"):
            raise ValueError(f"unknown objective '{objective}' (mux | discrete)")
        self.eval_loops = [int(k) for k in (eval_loops or [])]
        self.objective = objective

    def _trace_acc(self, ref_logits, x) -> float:
        trace = x["trace"]
        mask = x.get("trace_mask")
        mask = (trace != self.pad_id) if mask is None else mask.bool()
        pred = ref_logits[:, : trace.shape[1]].argmax(-1)
        return ((pred == trace) & mask).sum().item() / max(int(mask.sum()), 1)

    def discrete_loss(self, x: dict):
        """Stage 1: next-token prediction on the written-out trace and the
        answer, given the question (the ref term alone, no latent loop)."""
        logits, _ = self.forward_discrete(x["q"], x["trace"], x["answer"])
        eot = torch.full((x["q"].shape[0], 1), self.eot_id, dtype=torch.long, device=x["q"].device)
        targets = torch.cat([x["trace"], eot, x["answer"]], dim=1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.pad_id)
        with torch.no_grad():
            La = x["answer"].shape[1]
            ans_ok = (logits[:, -La].argmax(-1) == x["answer"][:, 0]).float().mean().item()
            parts = {"ref": loss.item(), "ref_answer_acc": ans_ok, "trace_acc": self._trace_acc(logits, x)}
        return loss, parts

    def training_loss(self, x: dict, y=None):
        if self.objective == "discrete":
            return self.discrete_loss(x)
        K = self.sample_n_latents()
        bptt = self.cfg.loop_bptt if self.training else None
        loss, parts = self.mux_loss(
            x["q"], x.get("spans"), x.get("span_mask"), x["trace"], x["answer"], n_latents=K, bptt=bptt
        )
        parts["loops"] = K
        if not self.training:
            with torch.no_grad():
                parts["trace_acc"] = self._trace_acc(self.forward_discrete(x["q"], x["trace"], x["answer"])[0], x)
            for k in self.eval_loops:
                with torch.no_grad():
                    _, logits, _ = self.forward_continuous(x["q"], x["answer"], n_latents=k)
                parts[f"acc_k{k}"] = (logits[:, 0].argmax(-1) == x["answer"][:, 0]).float().mean().item()
        return loss, parts
