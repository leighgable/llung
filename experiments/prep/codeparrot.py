#!/usr/bin/env python3
"""prep/codeparrot.py -- a CodeParrot subset + warm-start weights, offline.

Runs on the HOST: it needs the network, which the training sandboxes
do not have. It streams only as many documents as the token budget
needs (no full shard download), tokenizes them with codetok
(experiments/codetok: CodeParrot's BPE + special tokens; plain text
never maps to a special id), and writes the `tokens` task layout
(core/tasks.py:TokenStream):

    <out>/train.bin              uint16 token ids, docs joined by <|endoftext|>
    <out>/val.bin                from the held-out codeparrot-clean-valid split
    <out>/meta.json              dtype, vocab_size, eot id, tokenizer + sha256
    <out>/codeparrot-small.bin   GPT-2 weights (--weights), for init.checkpoint

    python experiments/prep/codeparrot.py --train-tokens 20_000_000

Then make <out> visible to the sandboxes, read-only:  /data add <out>

Inside a sandbox (no network) the agent fetches the files with the
`web_download` tool first (host-side, operator-approved, saved under
$LLUNG_RUNS_DIR/downloads/), then points --src at them:

    web_download url=<HF>/datasets/codeparrot/codeparrot-clean/resolve/main/file-000000000001.json.gz dest=codeparrot/file-000000000001.json.gz
    web_download url=<HF>/datasets/codeparrot/codeparrot-clean-valid/resolve/main/file-000000000054.json.gz dest=codeparrot/file-000000000054.json.gz
    web_download url=<HF>/codeparrot/codeparrot-small/resolve/main/pytorch_model.bin dest=codeparrot/pytorch_model.bin
    python experiments/prep/codeparrot.py --src $LLUNG_RUNS_DIR/downloads/codeparrot \
        --out $LLUNG_RUNS_DIR/data/codeparrot

One train shard holds ~100k files (well over 20M tokens); a missing
shard stops the run with the exact web_download call to make.

Note: codeparrot-small was trained on codeparrot-clean's train split,
so the warm-started model has seen the train subset. The val tokens
come from the separate valid split.
"""

from __future__ import annotations

import gzip
import io
import json
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Annotated, Iterator

import sys

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/
import codetok  # noqa: E402

HF = "https://huggingface.co"
MODEL = "codeparrot/codeparrot-small"
TRAIN = [f"{HF}/datasets/codeparrot/codeparrot-clean/resolve/main/file-{i:012d}.json.gz" for i in range(1, 54)]
VALID = [f"{HF}/datasets/codeparrot/codeparrot-clean-valid/resolve/main/file-000000000054.json.gz"]
DEFAULT_OUT = Path.home() / ".local/share/llung/data/codeparrot"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "llung-prep"}), timeout=60)


def _local(src: Path, url: str) -> Path:
    """The --src copy of `url` (its file name), or exit naming the download."""
    path = src / url.rsplit("/", 1)[-1]
    if not path.exists():
        sys.exit(f"missing {path}\n  fetch it (host-side): web_download "
                 f'{{"url": "{url}", "dest": "{src.name}/{path.name}"}}\n  then rerun.')
    return path


def download(url: str, dest: Path, src: Path | None = None) -> None:
    if dest.exists():
        typer.echo(f"have {dest.name}")
        return
    if src is not None:
        shutil.copyfile(_local(src, url), dest)
        typer.echo(f"copied {dest}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    with _open(url) as resp, part.open("wb") as f:
        total, done, t0 = int(resp.headers.get("Content-Length", 0)), 0, time.time()
        while block := resp.read(1 << 20):
            f.write(block)
            done += len(block)
            if done % (50 << 20) < (1 << 20):
                typer.echo(f"  {dest.name}: {done >> 20} / {total >> 20} MiB  ({done / 2**20 / max(time.time() - t0, 1e-3):.0f} MiB/s)")
    part.rename(dest)
    typer.echo(f"wrote {dest}")


def documents(urls: list[str], src: Path | None = None) -> Iterator[str]:
    """Stream `content` fields shard by shard, decompressing on the fly.
    With `src`, shards are read from local files (opened lazily, so only
    the shards the budget reaches must exist)."""
    for url in urls:
        opened = _local(src, url).open("rb") if src is not None else _open(url)
        with opened as resp, gzip.GzipFile(fileobj=resp) as gz:
            for line in io.TextIOWrapper(gz, encoding="utf-8"):
                yield json.loads(line)["content"]


def write_tokens(docs: Iterator[str], eot: int, budget: int, dest: Path) -> int:
    part = dest.with_suffix(".bin.part")
    n = n_docs = 0
    t0 = time.time()
    with part.open("wb") as f:
        batch: list[str] = []
        for doc in docs:
            batch.append(doc)
            if len(batch) < 256:
                continue
            for doc_ids in codetok.encode_batch(batch):
                ids = np.asarray(doc_ids + [eot], dtype=np.uint16)
                f.write(ids.tobytes())
                n += len(ids)
            n_docs += len(batch)
            batch = []
            typer.echo(f"  {dest.name}: {n:,} / {budget:,} tokens, {n_docs:,} docs, {n / max(time.time() - t0, 1e-3):,.0f} tok/s")
            if n >= budget:
                break
    part.rename(dest)
    return n


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Output data directory.")] = DEFAULT_OUT,
    train_tokens: Annotated[int, typer.Option(help="Train token budget (streams only what it needs).")] = 20_000_000,
    val_tokens: Annotated[int, typer.Option(help="Val token budget (held-out split).")] = 1_000_000,
    weights: Annotated[bool, typer.Option(help=f"Also fetch {MODEL}'s weights (~450 MB).")] = True,
    force: Annotated[bool, typer.Option(help="Re-tokenize even if train.bin/val.bin exist.")] = False,
    src: Annotated[Path | None, typer.Option(help="Read shards/weights from this dir (web_download'ed) instead of the network.")] = None,
):
    """Tokenize a CodeParrot subset (and fetch codeparrot-small) for the `tokens` task."""
    out = out.expanduser()
    src = src.expanduser() if src is not None else None
    out.mkdir(parents=True, exist_ok=True)
    eot, vocab, tok_sha = codetok.EOS, codetok.VOCAB_SIZE, codetok.sha256()
    assert vocab <= 65536, "uint16 storage needs vocab <= 65536"
    old = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    if old and old.get("tokenizer_sha256") != tok_sha and not force:
        sys.exit(f"{out} was tokenized with {old.get('tokenizer')} (sha256 "
                 f"{str(old.get('tokenizer_sha256'))[:12]}), not codetok {tok_sha[:12]}: rerun with --force")

    counts = {}
    for split, urls, budget in (("train", TRAIN, train_tokens), ("val", VALID, val_tokens)):
        dest = out / f"{split}.bin"
        if dest.exists() and not force:
            counts[split] = dest.stat().st_size // 2
            typer.echo(f"have {dest.name} ({counts[split]:,} tokens; --force to redo)")
            continue
        counts[split] = write_tokens(documents(urls, src), eot, budget, dest)

    if weights:
        download(f"{HF}/{MODEL}/resolve/main/pytorch_model.bin", out / "codeparrot-small.bin", src)

    meta = {
        "dtype": "uint16",
        "vocab_size": vocab,
        "eot_id": eot,
        "tokenizer": "codetok",
        "tokenizer_sha256": tok_sha,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "train_source": "codeparrot/codeparrot-clean (first shards, in file order)",
        "val_source": "codeparrot/codeparrot-clean-valid",
        "weights": "codeparrot-small.bin" if weights else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    typer.echo(f"\ndone: {out}\n{json.dumps(meta, indent=2)}\n\nnext: /data add {out}   (in the tui)")


if __name__ == "__main__":
    app()
