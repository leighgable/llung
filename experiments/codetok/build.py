#!/usr/bin/env python3
"""codetok/build.py -- regenerate codetok/tokenizer.json (host, network).

    python experiments/codetok/build.py

Takes codeparrot/codeparrot-small's byte-level BPE at a PINNED revision
(sha256-checked), appends the special tokens from codetok.SPECIALS at
ids 32768.., and writes experiments/codetok/tokenizer.json. The result
is committed, so sandboxes (no network) and every run use the same file.

Base ids are never changed: data tokenized with the upstream tokenizer
stays valid, and so does the upstream model's embedding matrix. To
introduce a new special token, rename a <|reserved_N|> slot in
codetok/__init__.py (the vocab size stays the same) and rebuild.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import codetok  # noqa: E402

REVISION = "e7e4f5d39319551a760f07c0e1035e379617c721"  # codeparrot/codeparrot-small
URL = f"https://huggingface.co/codeparrot/codeparrot-small/resolve/{REVISION}/tokenizer.json"
UPSTREAM_SHA256 = "6ce170cf04bbc296c76dae29de9b79f1a6d70bed6c5d7470e5ecc3f7a5696d45"


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=60).read()
    digest = hashlib.sha256(raw).hexdigest()
    if UPSTREAM_SHA256 and digest != UPSTREAM_SHA256:
        sys.exit(f"upstream tokenizer.json changed: sha256 {digest} != pinned {UPSTREAM_SHA256}")
    spec = json.loads(raw)
    vocab = spec["model"]["vocab"]
    if len(vocab) != codetok.BASE_VOCAB:
        sys.exit(f"expected a {codetok.BASE_VOCAB}-token base vocab, got {len(vocab)}")
    if [t["content"] for t in spec["added_tokens"]] != ["<|endoftext|>"] or vocab["<|endoftext|>"] != codetok.EOS:
        sys.exit(f"unexpected upstream special tokens: {spec['added_tokens']}")
    for i, name in enumerate(codetok.SPECIALS):
        spec["added_tokens"].append({
            "id": codetok.BASE_VOCAB + i, "content": name, "single_word": False,
            "lstrip": False, "rstrip": False, "normalized": False, "special": True,
        })
    out = HERE / "tokenizer.json"
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=None, separators=(",", ":")))
    tok = codetok.load(fresh=True)
    assert tok.get_vocab_size() == codetok.VOCAB_SIZE, tok.get_vocab_size()
    for i, name in enumerate(codetok.SPECIALS):
        assert tok.token_to_id(name) == codetok.BASE_VOCAB + i, name
    print(f"upstream sha256 {digest}")
    print(f"wrote {out}: vocab {codetok.VOCAB_SIZE} ({codetok.BASE_VOCAB} BPE + {len(codetok.SPECIALS)} special), "
          f"sha256 {codetok.sha256()}")


if __name__ == "__main__":
    main()
