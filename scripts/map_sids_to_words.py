"""Map a numeric semantic-id file to Hierarchical Textual IDs (word-codes).

Source-agnostic: takes ANY RecBole `item_sids.json` ({remapped_id: [codes...]})
produced by any tokenizer (LETTER, RQ-VAE, ...), optionally attaches each
codebook vector to a c-TF-IDF phrase, and
writes a word-code `item_sids.json` in the *same* format. Item ids are RecBole's
own remapped ids, so alignment is automatic:

    remapped id i  ==  row i of fields/items.jsonl  ==  key "i" of item_sids.json

The output must still be made globally-unique for TIGER via scripts/dedup_sids.py
(the same dedup step the numeric baseline uses).

Example (LETTER -> words, Sports):
    python scripts/map_sids_to_words.py \
        --sids  outputs/letter_sports/item_sids.json \
        --items data/processed/amazon2014_retrieval/Sports_and_Outdoors/fields/items.jsonl \
        --dataset Sports_and_Outdoors \
        --out   outputs/letter_sports/item_sids_words.json
    python scripts/dedup_sids.py outputs/letter_sports/item_sids_words.json \
                                 outputs/letter_sports/item_sids_words_dedup.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Reusable c-TF-IDF phrase-labeling utilities.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.sid_text_labels import label_and_build  # noqa: E402


def build_partition_preserving_ids(codes):
    """Injectively relabel every level-specific numeric code into a disjoint token range."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got shape {codes.shape}")
    mapped = np.empty_like(codes)
    mappings = []
    offset = 0
    for column in range(codes.shape[1]):
        values = sorted(int(value) for value in np.unique(codes[:, column]))
        mapping = {value: offset + index for index, value in enumerate(values)}
        mapped[:, column] = [mapping[int(value)] for value in codes[:, column]]
        mappings.append(mapping)
        offset += len(values)
    return mapped, mappings


def assert_prefix_partitions_preserved(source, mapped):
    """Verify prefix equality is equivalent before/after relabeling at every depth."""
    source = np.asarray(source)
    mapped = np.asarray(mapped)
    if source.shape != mapped.shape:
        raise AssertionError(f"partition-preserving relabel changed shape: {source.shape} -> {mapped.shape}")
    for depth in range(1, source.shape[1] + 1):
        forward = {}
        reverse = {}
        for old_row, new_row in zip(source[:, :depth], mapped[:, :depth], strict=True):
            old_prefix = tuple(int(value) for value in old_row)
            new_prefix = tuple(int(value) for value in new_row)
            if forward.setdefault(old_prefix, new_prefix) != new_prefix:
                raise AssertionError(f"original prefix split at depth {depth}: {old_prefix}")
            if reverse.setdefault(new_prefix, old_prefix) != old_prefix:
                raise AssertionError(f"original prefixes merged at depth {depth}: {new_prefix}")


def _load_codes(sids_path, drop_last_col):
    with open(sids_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    n = len(raw)
    keys = sorted(raw, key=int)
    if [int(k) for k in keys] != list(range(n)):
        raise ValueError(
            f"{sids_path}: keys must be a contiguous 0..N-1 range (got {keys[:3]}..{keys[-3:]})."
        )
    codes = np.asarray([[int(x) for x in raw[k]] for k in keys], dtype=np.int64)
    if drop_last_col:
        codes = codes[:, :-1]  # strip a pre-existing dedup digit
    return codes


def _load_texts(items_path, text_fields, n_expected):
    rows = []
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != n_expected:
        raise ValueError(
            f"{items_path} has {len(rows)} rows but the SID file has {n_expected} items. "
            "They must be the same length and in the same (remapped-id) order."
        )
    texts = []
    for r in rows:
        parts = [str(r.get(fld, "") or "") for fld in text_fields]
        texts.append(" . ".join(p for p in parts if p))
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sids", required=True, help="numeric item_sids.json to relabel")
    ap.add_argument("--items", required=True, help="fields/items.jsonl for this dataset")
    ap.add_argument("--dataset", required=True, help="Category (e.g. Beauty / Sports_and_Outdoors / Toys_and_Games)")
    ap.add_argument("--out", required=True, help="output word-code item_sids.json")
    ap.add_argument("--text_fields", default="title,brand,categories",
                    help="comma-sep items.jsonl fields used as the labeling corpus (default: title,brand,categories)")
    ap.add_argument("--top_k", type=int, default=5, help="# ranked words joined into each codebook token key")
    ap.add_argument("--top_n", type=int, default=5, help="# words emitted per vector label (>= top_k)")
    ap.add_argument(
        "--residual_mode",
        choices=("item", "global", "none"),
        default="global",
        help="item/global: residual-text redaction between codebooks (RQ-VAE/LETTER/ETEG); "
             "none: flat independent labeling for parallel/OPQ stacks (RPG)",
    )
    ap.add_argument(
        "--label_context", choices=("global", "parent"), default="global",
        help="global: current code-class c-TF-IDF; parent: score words against sibling "
             "branches while preserving the same (depth, code) targets",
    )
    ap.add_argument("--min_word_count", type=int, default=3)
    ap.add_argument("--min_doc_freq", type=int, default=5)
    ap.add_argument("--drop_last_col", action="store_true",
                    help="strip the last code column (use if the input already carries a dedup digit)")
    ap.add_argument("--offset_columns", action="store_true",
                    help="shift each codebook into a DISJOINT token range (position-unique "
                         "embeddings, matching an offset numeric RQ-VAE / preference_discerning's "
                         "expand_id) AND bake in the dedup digit -> writes a final, globally-unique "
                         "dedup file directly (do NOT run dedup_sids.py after).")
    ap.add_argument("--partition_preserving", action="store_true",
                    help="opt-in exact C1: injectively relabel each (level, original code), preserving "
                         "every prefix partition and original tuple collision; requires --offset_columns "
                         "and does not append a new dedup digit")
    ap.add_argument("--token_text_out", default="",
                    help="if set (with --offset_columns), also write {token_id: phrase} JSON in the "
                         "codec token space (token = offset_word_id + 1), 1:1 per position-unique token. "
                         "Used to text-ground SID token embeddings (TRACE C1-text).")
    args = ap.parse_args()
    if args.partition_preserving and not args.offset_columns:
        ap.error("--partition_preserving requires --offset_columns")

    text_fields = [s.strip() for s in args.text_fields.split(",") if s.strip()]
    codes = _load_codes(args.sids, args.drop_last_col)
    n_items, n_codebooks = codes.shape
    texts = _load_texts(args.items, text_fields, n_items)
    codebook_size = max(256, int(codes.max()) + 1)

    print(f"[map] {args.dataset}: {n_items} items x {n_codebooks} codebooks "
          f"(values in [{codes.min()},{codes.max()}]); corpus fields={text_fields}")
    print(f"[map] config: residual_mode={args.residual_mode} label_context={args.label_context} "
          f"word_id_top_k={args.top_k} "
          f"top_n={args.top_n} min_word_count={args.min_word_count} min_doc_freq={args.min_doc_freq}")

    word_ids, all_labels, vec_top1, stats = label_and_build(
        codes, texts, args.dataset, codebook_size,
        min_word_count=args.min_word_count, top_n=args.top_n,
        min_doc_freq=args.min_doc_freq, word_id_top_k=args.top_k,
        residual_mode=args.residual_mode, label_context=args.label_context,
    )
    print(f"[map] stats: {stats}")

    # eyeball: first few items, numeric codes -> per-codebook word key
    print("[map] sample items (numeric -> word key per codebook):")
    for i in range(min(5, n_items)):
        keys = [vec_top1[c].get(int(codes[i, c]), "?") for c in range(n_codebooks)]
        print(f"    id {i}: {list(map(int, codes[i]))} -> {list(map(int, word_ids[i]))}  |  "
              + "  ||  ".join(keys))

    if args.partition_preserving:
        original_codes = codes.copy()
        word_ids, code_mappings = build_partition_preserving_ids(original_codes)
        assert_prefix_partitions_preserved(original_codes, word_ids)
        if args.token_text_out:
            token_text = {}
            for c, mapping in enumerate(code_mappings):
                for original_code, mapped_code in mapping.items():
                    token_text[mapped_code + 1] = vec_top1[c].get(original_code, "")
            os.makedirs(os.path.dirname(os.path.abspath(args.token_text_out)) or ".", exist_ok=True)
            with open(args.token_text_out, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in sorted(token_text.items())}, f)
            print(f"[map] wrote token_text {args.token_text_out}  "
                  f"({len(token_text)} original code classes -> phrases)")
        print(f"[map] partition_preserving: exact injective relabel; width={n_codebooks} "
              f"vocab={int(word_ids.max()) + 1}; no dedup digit added")
    elif args.offset_columns:
        # 1) tight, disjoint per-column offset -> each codebook gets its own token
        #    block, so the position-agnostic TIGER codec yields position-UNIQUE
        #    embeddings (fair vs an offset numeric RQ-VAE baseline).
        off = 0
        for c in range(n_codebooks):
            word_ids[:, c] += off
            off = int(word_ids[:, c].max()) + 1
        # token_id -> phrase (codec applies token = value + 1). Columns are now disjoint, so each
        # offset word id maps to exactly ONE phrase. Emitted BEFORE the dedup digit is appended
        # (the dedup counter carries no text). n_codebooks here is still the word-column count.
        if args.token_text_out:
            token_text = {}
            for i in range(n_items):
                for c in range(n_codebooks):
                    tok = int(word_ids[i, c]) + 1  # codec offset (value + 1)
                    token_text[tok] = vec_top1[c].get(int(codes[i, c]), "")
            os.makedirs(os.path.dirname(os.path.abspath(args.token_text_out)) or ".", exist_ok=True)
            with open(args.token_text_out, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in sorted(token_text.items())}, f)
            print(f"[map] wrote token_text {args.token_text_out}  ({len(token_text)} SID tokens -> phrases)")
        # 2) bake the dedup digit in its OWN block (also position-unique), so the
        #    file is globally-unique without scripts/dedup_sids.py.
        from collections import Counter
        seen = Counter()
        dd = np.zeros(n_items, dtype=np.int64)
        for i in range(n_items):
            t = tuple(int(x) for x in word_ids[i])
            dd[i] = seen[t]
            seen[t] += 1
        dd += off
        word_ids = np.concatenate([word_ids, dd[:, None]], axis=1)
        n_codebooks += 1
        maxdd = int(dd.max() - off)
        print(f"[map] offset_columns: disjoint per-codebook ranges + baked dedup "
              f"(max dedup counter={maxdd}); final width={n_codebooks} vocab~{int(word_ids.max())+1}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = {str(i): [int(x) for x in word_ids[i]] for i in range(n_items)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    if args.partition_preserving:
        print(f"[map] wrote {args.out}  (partition-preserving; all {n_codebooks} prefix depths verified)")
    elif args.offset_columns:
        uniq = len(set(map(tuple, word_ids.tolist())))
        print(f"[map] wrote {args.out}  (offset+dedup baked; unique_tuples={uniq}/{n_items}; "
              f"do NOT run dedup_sids.py)")
    else:
        print(f"[map] wrote {args.out}  (next: scripts/dedup_sids.py {args.out} <..._dedup.json>)")


if __name__ == "__main__":
    main()
