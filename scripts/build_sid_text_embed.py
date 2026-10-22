#!/usr/bin/env python
"""Text-ground SID token embeddings (TRACE C1-text).

Given a ``token_text.json`` ({token_id: "five word phrase"}) emitted by
map_sids_to_words.py --offset_columns --token_text_out (position-unique tokens, 1:1
token->phrase), encode each phrase with a local pretrained sentence-T5 and save a
per-token embedding table the TIGER model injects into its input/output embeddings.

Output (torch.save): {
    "embed":    float32 [V, d_text]  (V = max token id + 1; text-less rows are zero),
    "has_text": bool    [V],
    "dim":      int (d_text, e.g. 768),
    "model":    str,
}
Offline: needs sentence-transformers/sentence-t5-base in the HF cache (HF_HUB_OFFLINE=1).

Usage:
    python scripts/build_sid_text_embed.py \
        --token_text outputs/tiger_toys/item_sids_words_flatoffset.token_text.json \
        --out        outputs/tiger_toys/sid_text_embed.pt
"""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token_text", required=True, help="{token_id: phrase} JSON from map_sids_to_words")
    ap.add_argument("--out", required=True, help="output .pt embedding table")
    ap.add_argument("--model", default="sentence-transformers/sentence-t5-base",
                    help="local sentence-transformers model (offline)")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    with open(args.token_text, encoding="utf-8") as f:
        tok2phrase = {int(k): (v or "").strip() for k, v in json.load(f).items()}
    if not tok2phrase:
        raise SystemExit(f"{args.token_text}: empty token_text map")

    vocab = max(tok2phrase) + 1
    # Encode the UNIQUE phrases once (many tokens repeat words; dedup keeps it cheap).
    ids = sorted(t for t, p in tok2phrase.items() if p)
    phrases = [tok2phrase[t] for t in ids]
    uniq = sorted(set(phrases))
    print(f"[embed] {len(tok2phrase)} tokens ({len(ids)} with text, {len(uniq)} unique phrases); vocab={vocab}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    vecs = model.encode(uniq, batch_size=args.batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    dim = int(vecs.shape[1])
    phrase2vec = {p: vecs[i] for i, p in enumerate(uniq)}

    embed = torch.zeros((vocab, dim), dtype=torch.float32)
    has_text = torch.zeros((vocab,), dtype=torch.bool)
    for t in ids:
        embed[t] = torch.from_numpy(phrase2vec[tok2phrase[t]])
        has_text[t] = True

    torch.save({"embed": embed, "has_text": has_text, "dim": dim, "model": args.model}, args.out)
    print(f"[embed] wrote {args.out}  shape={tuple(embed.shape)} has_text={int(has_text.sum())}/{vocab} dim={dim}")


if __name__ == "__main__":
    main()
