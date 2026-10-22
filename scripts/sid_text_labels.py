"""
c-TF-IDF SID phrase labeling and legacy word-id construction.

Given the numeric semantic-id assignments produced by the RQ-VAE (a
``(n_items, n_codebooks)`` int array, one column per codebook) and a labeling
corpus of item metadata sentences, attach human-readable *words* to every used
codebook vector via a class-based TF-IDF pass (BERTopic-style), using the
residual-text scheme so the codebooks describe complementary facets.

The top-1 word of each item's assigned vector (per codebook) is then remapped
to a dense per-codebook integer in ``[0, codebook_size)`` and written out as a
word-mapped id pickle in **exactly the same format as the numeric semantic-id
pickle**. That means the existing GenRet decoder / dataloader consume it with no
changes: two vectors whose top-1 word is identical collapse to the same token,
tying together items the numeric ids keep apart, while the dataloader's dedup
column keeps every item individually retrievable.

The reusable phrase-labeling functions support lexical C1PP artifact construction.
"""

import argparse
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np

try:  # package import (e.g. ``from scripts.sid_text_labels import ...``)
    from .sid_text_clean import clean_texts
except ImportError:  # direct script execution
    from sid_text_clean import clean_texts


# --------------------------------------------------------------------------- #
# c-TF-IDF scoring (§4.1 + §4.2 of the porting spec)
# --------------------------------------------------------------------------- #
def _score_codebook(codes_col, tokens_cur, min_word_count, top_n):
    """Score words for every used vector of a single codebook.

    Args:
        codes_col: (n_items,) int, the vector index assigned to each item.
        tokens_cur: list of per-item word lists (current residual tokens).
        min_word_count: strict tf threshold, relaxed downward per-class until
            a non-empty label is produced.
        top_n: number of words to keep per vector.

    Returns:
        dict {vector_index: [words ranked best-first]} for every used vector.
    """
    # corpus document-frequency (== term-frequency since words are deduped)
    total_freq = Counter()
    for words in tokens_cur:
        total_freq.update(words)

    class_items = defaultdict(list)
    for i, v in enumerate(codes_col):
        class_items[int(v)].append(i)

    class_token_sums = [
        sum(len(tokens_cur[i]) for i in items) for items in class_items.values()
    ]
    avg_tokens = float(np.mean(class_token_sums)) if class_token_sums else 0.0

    labels = {}
    for v, items in class_items.items():
        counts = Counter()
        for i in items:
            counts.update(tokens_cur[i])
        class_size = len(items)

        scored = {}
        for eff_min in range(min(min_word_count, class_size), 0, -1):
            scored = {
                w: tf * math.log(1.0 + avg_tokens / total_freq[w])
                for w, tf in counts.items()
                if tf >= eff_min
            }
            if scored:
                break

        # rank by score desc, then word asc for determinism
        top = sorted(scored, key=lambda w: (-scored[w], w))[:top_n]
        labels[v] = top
    return labels


def build_labels_flat(codes, tokens, n_codebooks, min_word_count=3, top_n=5):
    """Flat / independent labeling for parallel (OPQ-style) codebooks.

    Each codebook is scored on the **full** cleaned item text. No residual
    redaction between digits — complementary to :func:`build_labels_residual`.

    Returns dict {codebook_index (0-based): {vector: [words]}}.
    """
    tokens_full = [list(t) for t in tokens]
    return {
        c: _score_codebook(codes[:, c], tokens_full, min_word_count, top_n)
        for c in range(n_codebooks)
    }


def build_labels_parent_contrastive(
    codes, tokens, n_codebooks, min_word_count=3, top_n=5
):
    """Label each depth/code using words that distinguish it from siblings.

    The decoder still receives one token row per ``(depth, code)``.  For every
    parent prefix where that code occurs, words are rewarded when frequent in
    the child and rare among the other children of the same parent.  Scores
    are summed across parents, yielding one parent-aware phrase without
    changing any SID partition or target.
    """
    codes = np.asarray(codes)
    all_labels = {}
    for depth in range(n_codebooks):
        parent_items = defaultdict(list)
        edge_items = defaultdict(list)
        code_items = defaultdict(list)
        for item, row in enumerate(codes):
            parent = tuple(int(value) for value in row[:depth])
            code = int(row[depth])
            parent_items[parent].append(item)
            edge_items[(parent, code)].append(item)
            code_items[code].append(item)

        parent_counts = {}
        for parent, items in parent_items.items():
            counts = Counter()
            for item in items:
                counts.update(tokens[item])
            parent_counts[parent] = counts

        scores = defaultdict(Counter)
        total_counts = defaultdict(Counter)
        for code, items in code_items.items():
            for item in items:
                total_counts[code].update(tokens[item])

        for (parent, code), items in edge_items.items():
            child = Counter()
            for item in items:
                child.update(tokens[item])
            sibling_size = len(parent_items[parent]) - len(items)
            for word, tf in child.items():
                sibling_df = parent_counts[parent][word] - tf
                # Smoothed sibling-conditioned IDF. A word receives more
                # weight when it is common in this child but absent from the
                # alternative branches under the same parent.
                contrast = math.log(1.0 + (sibling_size + 1.0) / (sibling_df + 1.0))
                scores[code][word] += tf * contrast

        labels = {}
        for code in sorted(code_items):
            counts = total_counts[code]
            ranked = []
            class_size = len(code_items[code])
            for eff_min in range(min(min_word_count, class_size), 0, -1):
                ranked = [word for word, tf in counts.items() if tf >= eff_min]
                if ranked:
                    break
            labels[code] = sorted(
                ranked, key=lambda word: (-scores[code][word], word)
            )[:top_n]
        all_labels[depth] = labels
    return all_labels


def build_labels_residual(
    codes, tokens, n_codebooks, min_word_count=3, top_n=5, residual_mode="item"
):
    """Residual-text labeling: score codebook c, then redact claimed words
    before scoring codebook c+1 (§4.3).

    ``residual_mode``:
      * ``item`` (default): drop, for each item, only the words in the label of
        *that item's* assigned vector at codebook c.
      * ``global``: drop every word that appears in *any* codebook-c label from
        *all* items' residual texts.

    For parallel/OPQ stacks use :func:`build_labels_flat` instead (via
    ``residual_mode='none'`` in :func:`label_and_build`).

    Returns dict {codebook_index (0-based): {vector: [words]}}.
    """
    if residual_mode not in ("item", "global"):
        raise ValueError(
            f"build_labels_residual residual_mode must be 'item' or 'global', "
            f"got {residual_mode!r} (use build_labels_flat / residual_mode='none' for OPQ)"
        )
    tokens_cur = [list(t) for t in tokens]
    all_labels = {}
    for c in range(n_codebooks):
        labels = _score_codebook(codes[:, c], tokens_cur, min_word_count, top_n)
        all_labels[c] = labels
        if residual_mode == "global":
            claimed = {w for ws in labels.values() for w in ws}
            tokens_cur = [[w for w in toks if w not in claimed] for toks in tokens_cur]
        else:
            # per-item: drop words claimed by THIS item's assigned vector at c
            top_sets = {v: set(ws) for v, ws in labels.items()}
            tokens_cur = [
                [w for w in toks if w not in top_sets.get(int(codes[i, c]), ())]
                for i, toks in enumerate(tokens_cur)
            ]
    return all_labels


def build_word_ids(codes, all_labels, n_codebooks, codebook_size, word_id_top_k=1):
    """Map each item's assigned-vector label words to a dense per-codebook id.

    ``word_id_top_k`` controls how many ranked label words form the token key
    (1 = top-1 only; 5 = join the top-5 into one atomic key). Vectors with an
    empty label get a unique pseudo-word so they stay distinct.
    Returns (word_ids ``(n_items, n_codebooks)`` int64, vec_top1 dict).
    """
    n_items = len(codes)
    word_ids = np.zeros((n_items, n_codebooks), dtype=np.int64)
    vec_top1_all = {}
    k = max(1, int(word_id_top_k))
    for c in range(n_codebooks):
        labels = all_labels[c]
        vec_top1 = {}
        for v, words in labels.items():
            if words:
                vec_top1[v] = " ".join(words[:k])
            else:
                vec_top1[v] = f"__code_{c}_{v}"
        word2local = {}
        for i in range(n_items):
            v = int(codes[i, c])
            word = vec_top1.get(v, f"__code_{c}_{v}")
            if word not in word2local:
                word2local[word] = len(word2local)
            word_ids[i, c] = word2local[word]
        if len(word2local) > codebook_size:
            raise ValueError(
                f"Codebook {c + 1} needs {len(word2local)} distinct word tokens "
                f"but codebook_size is {codebook_size}; the numeric-id vocab "
                f"layout cannot represent this. Reduce vocabulary or widen the "
                f"decoder vocab."
            )
        vec_top1_all[c] = vec_top1
    return word_ids, vec_top1_all


def _top1(words_str):
    """First (highest-ranked) word of a space-joined label string."""
    if not isinstance(words_str, str):
        return None
    toks = words_str.split()
    return toks[0] if toks else None


def _title_key(text):
    """Normalized Title-only join key (robust to punctuation / HTML entities /
    brand-field formatting differences between codebases)."""
    import html

    s = html.unescape(str(text)).lower()
    s = s.split("brand:")[0].split("categories:")[0].replace("title:", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _align_rows(df, id2meta):
    """Map each of this codebase's items (id 1..n, ascending) to a row of the
    external labels ``df``. Uses positional alignment when the CSV shares this
    codebase's item ordering, else a normalized-title join (handles reordered
    datasets like Toys). Returns (chosen_row_per_item, n_items, mode, n_unmatched).
    """
    this_items = sorted(id2meta.items(), key=lambda kv: int(kv[0]))  # (id, text)
    n_items = len(this_items)
    this_keys = [_title_key(v) for _, v in this_items]
    csv_keys = [_title_key(t) for t in df["text"].tolist()]

    # positional check: csv row i vs this item i+1
    m = min(n_items, len(csv_keys))
    pos_match = sum(1 for i in range(m) if csv_keys[i] == this_keys[i]) / max(m, 1)
    if len(csv_keys) == n_items and pos_match > 0.9:
        return list(range(n_items)), n_items, f"positional({pos_match:.3f})", 0

    # title join, preferring an unused CSV row so duplicate titles pair 1:1
    by_key = {}
    for ri, k in enumerate(csv_keys):
        by_key.setdefault(k, []).append(ri)
    used, chosen, n_unmatched = set(), [], 0
    for k in this_keys:
        cands = by_key.get(k, [])
        pick = next((ri for ri in cands if ri not in used), cands[0] if cands else None)
        if pick is None:
            n_unmatched += 1
        else:
            used.add(pick)
        chosen.append(pick)
    return chosen, n_items, f"title-join(pos={pos_match:.3f})", n_unmatched


def build_word_ids_from_labels(item_sem_ids_csv, codebook_size, n_codebooks=3,
                               id2meta=None, word_id_top_k=1):
    """Build word ids from a pre-computed ``item_sem_ids_residual.csv``.

    Consumes the *per-item words* from an external Stage-C run (columns
    ``l1_words, l2_words, l3_words`` plus ``item_id`` and ``sem_id``). The first
    ``word_id_top_k`` words of each codebook column become the token key; items
    sharing a vector share the same ``l*_words`` and therefore the same token --
    identical to labeling the codebooks directly.

    Alignment to *this* codebase's items:
      * ``id2meta`` given -> map by item identity (position or title join, auto).
      * ``id2meta`` None  -> assume row ``i`` == item ``i+1`` (positional).

    Empty labels fall back to a per-vector pseudo-word (from ``sem_id``); the rare
    unmatched item gets a per-item pseudo-word. Neither inflates the vocabulary
    materially.
    """
    import pandas as pd

    df = pd.read_csv(item_sem_ids_csv).sort_values("item_id").reset_index(drop=True)
    word_cols = [f"l{c + 1}_words" for c in range(n_codebooks)]
    for col in word_cols + ["item_id", "sem_id"]:
        assert col in df.columns, f"{item_sem_ids_csv} missing required column '{col}'"

    if id2meta is not None:
        chosen, n_items, mode, n_unmatched = _align_rows(df, id2meta)
    else:
        chosen, n_items, mode, n_unmatched = list(range(len(df))), len(df), "positional", 0

    sem_parts = df["sem_id"].astype(str).str.split("-").tolist()
    col_vals = [df[word_cols[c]].tolist() for c in range(n_codebooks)]
    word_ids = np.zeros((n_items, n_codebooks), dtype=np.int64)
    k = max(1, int(word_id_top_k))
    for c in range(n_codebooks):
        word2local = {}
        for i, ri in enumerate(chosen):
            if ri is None:
                word = f"__unmatched_{c}_{i}"
            else:
                raw = col_vals[c][ri]
                if isinstance(raw, str) and raw.strip():
                    word = " ".join(raw.split()[:k])
                else:
                    word = None
                if word is None:
                    vec = sem_parts[ri][c] if len(sem_parts[ri]) > c else ri
                    word = f"__code_{c}_{vec}"
            if word not in word2local:
                word2local[word] = len(word2local)
            word_ids[i, c] = word2local[word]
        if len(word2local) > codebook_size:
            raise ValueError(
                f"Codebook {c + 1} needs {len(word2local)} distinct word tokens "
                f"but codebook_size is {codebook_size}."
            )

    n_unique, max_collisions = check_dedup_capacity(word_ids, codebook_size)
    stats = {
        "n_items": n_items,
        "n_codebooks": n_codebooks,
        "alignment": mode,
        "n_unmatched_items": n_unmatched,
        "word_id_top_k": k,
        "unique_word_triples": n_unique,
        "max_word_triple_collisions": max_collisions,
        "distinct_word_tokens_per_codebook": [
            len(set(word_ids[:, c].tolist())) for c in range(n_codebooks)
        ],
    }
    return word_ids, stats


def check_dedup_capacity(word_ids, codebook_size):
    """The GenRet dataloader appends a per-collision dedup counter that must fit
    the same ``codebook_size`` slot block. Verify word-triples don't collide
    more than ``codebook_size`` times (else the dedup token would overflow into
    the user/EOS token range).
    """
    counts = Counter(map(tuple, word_ids.tolist()))
    max_collisions = max(counts.values()) if counts else 0
    n_unique = len(counts)
    if max_collisions > codebook_size:
        raise ValueError(
            f"Max word-triple collisions {max_collisions} exceeds codebook_size "
            f"{codebook_size}; the dataloader's dedup column would overflow. "
            f"Consider a larger top_n / min_word_count or a healthier RQ-VAE."
        )
    return n_unique, max_collisions


# --------------------------------------------------------------------------- #
# End-to-end build
# --------------------------------------------------------------------------- #
def label_and_build(codes, texts, dataset, codebook_size, min_word_count=3,
                    top_n=5, min_doc_freq=5, word_id_top_k=1,
                    residual_mode="item", label_context="global"):
    """Run the full Stage-C pipeline in memory.

    Args:
        codes: ``(n_items, n_codebooks)`` int array of numeric assignments.
        texts: list of raw metadata sentences, row-aligned to ``codes``.
        dataset: dataset name (for domain-word removal).
        word_id_top_k: how many label words to join into each ID token key.
        residual_mode: ``item`` / ``global`` dispatch to
            :func:`build_labels_residual`; ``none`` dispatches to
            :func:`build_labels_flat` (OPQ / parallel codebooks).
    Returns:
        (word_ids, all_labels, vec_top1, stats).
    """
    codes = np.asarray(codes)
    assert codes.ndim == 2, f"codes must be 2-D, got shape {codes.shape}"
    assert len(texts) == len(codes), (
        f"corpus/codes length mismatch: {len(texts)} texts vs {len(codes)} codes"
    )
    if residual_mode not in ("item", "global", "none"):
        raise ValueError(
            f"residual_mode must be 'item', 'global', or 'none', got {residual_mode!r}"
        )
    if label_context not in ("global", "parent"):
        raise ValueError(f"label_context must be 'global' or 'parent', got {label_context!r}")
    if label_context == "parent" and residual_mode != "none":
        raise ValueError("parent label_context currently requires residual_mode='none'")
    n_codebooks = codes.shape[1]

    tokens = clean_texts(texts, dataset, min_doc_freq=min_doc_freq)
    n_empty = sum(1 for t in tokens if not t)

    if label_context == "parent":
        all_labels = build_labels_parent_contrastive(
            codes, tokens, n_codebooks, min_word_count=min_word_count, top_n=top_n,
        )
    elif residual_mode == "none":
        all_labels = build_labels_flat(
            codes, tokens, n_codebooks, min_word_count=min_word_count, top_n=top_n,
        )
    else:
        all_labels = build_labels_residual(
            codes, tokens, n_codebooks, min_word_count=min_word_count, top_n=top_n,
            residual_mode=residual_mode,
        )
    word_ids, vec_top1 = build_word_ids(
        codes, all_labels, n_codebooks, codebook_size, word_id_top_k=word_id_top_k
    )
    n_unique, max_collisions = check_dedup_capacity(word_ids, codebook_size)

    stats = {
        "n_items": len(codes),
        "n_codebooks": n_codebooks,
        "n_empty_processed_texts": n_empty,
        "word_id_top_k": word_id_top_k,
        "residual_mode": residual_mode,
        "label_context": label_context,
        "unique_word_triples": n_unique,
        "max_word_triple_collisions": max_collisions,
        "distinct_word_tokens_per_codebook": [
            len({vec_top1[c][int(v)] for v in np.unique(codes[:, c])})
            for c in range(n_codebooks)
        ],
    }
    return word_ids, all_labels, vec_top1, stats


def write_outputs(word_ids, all_labels, texts, codes, out_pkl, labels_csv=None,
                  item_ids_csv=None):
    """Persist the word-id pickle (+ optional human-readable CSVs)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_pkl)), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(word_ids, f)

    if labels_csv is not None:
        import pandas as pd

        rows = []
        for c, labels in all_labels.items():
            for v, words in sorted(labels.items()):
                rows.append(
                    {"codebook": c + 1, "vector": v,
                     "n_words": len(words), "words": " ".join(words)}
                )
        pd.DataFrame(rows).to_csv(labels_csv, index=False)

    if item_ids_csv is not None:
        import pandas as pd

        n_cb = codes.shape[1]
        vec_top1 = {}
        for c, labels in all_labels.items():
            vec_top1[c] = {v: (ws[0] if ws else f"__code_{c}_{v}")
                           for v, ws in labels.items()}
        rows = []
        for i in range(len(codes)):
            row = {
                "item_id": i + 1,
                "sem_id": "-".join(str(int(codes[i, c])) for c in range(n_cb)),
                "word_id": "-".join(str(int(word_ids[i, c])) for c in range(n_cb)),
            }
            for c in range(n_cb):
                row[f"l{c + 1}_word"] = vec_top1[c].get(int(codes[i, c]), "")
            row["text"] = texts[i]
            rows.append(row)
        pd.DataFrame(rows).to_csv(item_ids_csv, index=False)


# --------------------------------------------------------------------------- #
# Integration hook -- called from GenRet/training.py when id_mode == "words".
# Derives every path from the training config; caches the built pickle.
# --------------------------------------------------------------------------- #
def _load_id2meta(config):
    """Load this codebase's item-id -> metadata sentence map (the alignment
    anchor). Row order of the codes/word-id pickle is item id ascending."""
    dataset = config["name"]
    features_used = "_".join(config["features_needed"])
    suffix = "_id2meta_prompted.json" if config.get("use_prompt") else "_id2meta.json"
    meta_path = os.path.join(
        "ID_generation", "preprocessing", "processed",
        f"{dataset}_{features_used}{suffix}",
    )
    with open(meta_path, "r") as f:
        return json.load(f)


def build_word_id_pkl(config, numeric_pkl_path, codes_pkl=None, labels_csv=None,
                      top_n=5, min_word_count=3, min_doc_freq=5, tag="residual",
                      rebuild=False, word_id_top_k=1, residual_mode="item"):
    """Produce (and cache) a word-mapped id pickle next to the numeric one.

    Three sources, in priority order:
      * ``labels_csv``: a pre-computed ``item_sem_ids_residual.csv`` (per-item
        words from an external Stage-C run) -- used as-is, no codes/RQ-VAE.
      * ``codes_pkl``: an existing numeric semantic-id pickle -- labeled here.
      * else ``numeric_pkl_path`` (the freshly produced pickle) -- labeled here.

    Returns the path to the word-mapped id pickle (same
    ``(n_items, n_codebooks)`` int format, consumed by the dataloader unchanged).
    """
    codebook_size = config["RQ-VAE"]["code_book_size"]
    n_codebooks = config["RQ-VAE"]["num_layers"]

    out_pkl = numeric_pkl_path.replace(".pkl", f"_words_{tag}.pkl")
    out_labels_csv = numeric_pkl_path.replace(".pkl", f"_codebook_labels_{tag}.csv")
    item_ids_csv = numeric_pkl_path.replace(".pkl", f"_item_word_ids_{tag}.csv")

    if os.path.exists(out_pkl) and not rebuild:
        print(f"[sid_text_labels] reusing cached word ids: {out_pkl}")
        return out_pkl

    # --- Source 1: a pre-computed per-item labels CSV --------------------------
    if labels_csv is not None:
        print(f"[sid_text_labels] building word ids from labels CSV {labels_csv}")
        id2meta = _load_id2meta(config)
        word_ids, stats = build_word_ids_from_labels(
            labels_csv, codebook_size, n_codebooks=n_codebooks, id2meta=id2meta,
            word_id_top_k=word_id_top_k,
        )
        os.makedirs(os.path.dirname(os.path.abspath(out_pkl)), exist_ok=True)
        with open(out_pkl, "wb") as f:
            pickle.dump(word_ids, f)
        print(f"[sid_text_labels] built word ids for {config['name']}: {stats}")
        print(f"[sid_text_labels] wrote {out_pkl}")
        return out_pkl

    # --- Sources 2/3: label this codebase's numeric codes ---------------------
    dataset = config["name"]
    features_used = "_".join(config["features_needed"])
    source_codes = codes_pkl or numeric_pkl_path
    print(f"[sid_text_labels] reading numeric codes from {source_codes}")
    with open(source_codes, "rb") as f:
        codes = np.asarray(pickle.load(f))

    # row i of `codes` <-> the i-th item id in ascending order (matches how the
    # RQ-VAE embeddings were built in ID_generation/generate_id.py).
    id2meta = _load_id2meta(config)
    texts = [v for _, v in sorted(id2meta.items(), key=lambda kv: int(kv[0]))]

    word_ids, all_labels, _, stats = label_and_build(
        codes, texts, dataset, codebook_size,
        min_word_count=min_word_count, top_n=top_n, min_doc_freq=min_doc_freq,
        word_id_top_k=word_id_top_k, residual_mode=residual_mode,
    )
    write_outputs(word_ids, all_labels, texts, codes, out_pkl,
                  labels_csv=out_labels_csv, item_ids_csv=item_ids_csv)
    print(f"[sid_text_labels] built word ids for {dataset}: {stats}")
    print(f"[sid_text_labels] wrote {out_pkl}")
    return out_pkl


# --------------------------------------------------------------------------- #
# Standalone Stage-C CLI (offline usage, independent of a training run)
# --------------------------------------------------------------------------- #
def _load_codes(codes_path):
    with open(codes_path, "rb") as f:
        return np.asarray(pickle.load(f))


def main():
    ap = argparse.ArgumentParser(description="Hierarchical Textual IDs -- Stage C")
    ap.add_argument("--codes", required=True,
                    help="numeric semantic-id pickle, shape (n_items, n_codebooks)")
    ap.add_argument("--id2meta", required=True,
                    help="JSON mapping item id -> metadata sentence")
    ap.add_argument("--dataset", required=True, help="dataset name (e.g. Beauty)")
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--out_pkl", required=True,
                    help="output word-mapped id pickle (numeric-id format)")
    ap.add_argument("--labels_csv", default=None)
    ap.add_argument("--item_ids_csv", default=None)
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--min_word_count", type=int, default=3)
    ap.add_argument("--min_doc_freq", type=int, default=5)
    ap.add_argument("--residual_mode", choices=("item", "global", "none"), default="item")
    ap.add_argument("--word_id_top_k", type=int, default=1)
    args = ap.parse_args()

    codes = _load_codes(args.codes)
    with open(args.id2meta, "r") as f:
        id2meta = json.load(f)
    texts = [v for _, v in sorted(id2meta.items(), key=lambda kv: int(kv[0]))]

    word_ids, all_labels, _, stats = label_and_build(
        codes, texts, args.dataset, args.codebook_size,
        min_word_count=args.min_word_count, top_n=args.top_n,
        min_doc_freq=args.min_doc_freq, residual_mode=args.residual_mode,
        word_id_top_k=args.word_id_top_k,
    )
    write_outputs(word_ids, all_labels, texts, codes, args.out_pkl,
                  labels_csv=args.labels_csv, item_ids_csv=args.item_ids_csv)
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out_pkl}")


if __name__ == "__main__":
    main()
