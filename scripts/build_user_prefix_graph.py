#!/usr/bin/env python
"""Export a user <-> exact-SID-prefix graph for LightGCN.

Each train interaction with an item SID ``[q1, ..., qL]`` connects its user to
the exact prefix nodes ``[q1]``, ``[q1,q2]``, ... at informative depths.  Thus
``(depth=2, code=7)`` beneath different parents remains distinct.  The output
uses the existing LightGCN ``train.txt`` contract and is isolated under
``data/user_prefix_graph/<family>_<tag>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TAG_TO_CATEGORY = {'beauty': 'Beauty', 'sports': 'Sports_and_Outdoors', 'toys': 'Toys_and_Games', 'musical': 'Musical_Instruments', 'grocery': 'Grocery_and_Gourmet_Food', 'pet_supplies': 'Pet_Supplies', 'video_games': 'Video_Games'}

def _load_jsonl_ids(path: Path, key: str) -> list[str]:
    ids: list[str] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)[key]))
    return ids

def _train_histories(interactions_path, *, asin_to_item, user_to_id, hold):
    by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with interactions_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            asin = str(row['item_id'])
            if asin not in asin_to_item:
                continue
            by_user[str(row['user_id'])].append((int(row['timestamp']), asin))
    histories: dict[int, list[int]] = {}
    for (raw_user, events) in by_user.items():
        if raw_user not in user_to_id:
            continue
        events.sort(key=lambda x: x[0])
        items = [asin_to_item[a] for (_, a) in events]
        if len(items) <= hold:
            continue
        seen: set[int] = set()
        uniq: list[int] = []
        for item_id in items[:len(items) - hold]:
            if item_id not in seen:
                seen.add(item_id)
                uniq.append(item_id)
        if uniq:
            histories[user_to_id[raw_user]] = uniq
    return histories



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--family", default="letter")
    parser.add_argument("--sid-file", type=Path, default=None)
    parser.add_argument(
        "--positions",
        default="",
        help="comma-separated informative SID positions; default drops positions with <=2 values",
    )
    parser.add_argument("--valid-holdout", type=int, default=1)
    parser.add_argument("--test-holdout", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _display_path(path: Path) -> str:
    """Use a portable project-relative path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    if args.tag not in TAG_TO_CATEGORY:
        raise SystemExit(f"Unknown tag {args.tag!r}; choose from {sorted(TAG_TO_CATEGORY)}")

    category = TAG_TO_CATEGORY[args.tag]
    processed = ROOT / "data" / "processed" / "amazon2014_retrieval" / category / "sentence"
    sid_path = args.sid_file or (
        ROOT / "outputs" / f"{args.family}_{args.tag}" / "item_sids_words_flatoffset_pp_dedup.json"
    )
    items_path = processed / "items.jsonl"
    users_path = processed / "users.jsonl"
    interactions_path = processed / "interactions.jsonl"
    for path in (sid_path, items_path, users_path, interactions_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing input: {path}")

    asin_list = _load_jsonl_ids(items_path, "item_id")
    user_list = _load_jsonl_ids(users_path, "user_id")
    asin_to_item = {asin: index for index, asin in enumerate(asin_list)}
    user_to_id = {user: index for index, user in enumerate(user_list)}
    item_codes = {int(item): tuple(map(int, code)) for item, code in json.load(sid_path.open()).items()}
    width = len(next(iter(item_codes.values())))
    if any(len(code) != width for code in item_codes.values()):
        raise ValueError("SID tuples have inconsistent widths")

    if args.positions.strip():
        positions = [int(value) for value in args.positions.split(",") if value.strip()]
    else:
        positions = [
            pos
            for pos in range(width)
            if len({code[pos] for code in item_codes.values()}) > 2
        ]
    if not positions:
        raise ValueError("No informative positions selected")
    if any(pos < 0 or pos >= width for pos in positions):
        raise ValueError(f"positions must be in [0, {width - 1}]")

    histories = _train_histories(
        interactions_path,
        asin_to_item=asin_to_item,
        user_to_id=user_to_id,
        hold=int(args.valid_holdout) + int(args.test_holdout),
    )
    if not histories:
        raise ValueError("No train histories after holdout")
    user_nodes = {user: node for node, user in enumerate(sorted(histories))}

    edge_weight: dict[tuple[int, tuple[int, ...]], int] = defaultdict(int)
    prefix_count: Counter[tuple[int, ...]] = Counter()
    for user, items in histories.items():
        user_node = user_nodes[user]
        for item in items:
            code = item_codes.get(item)
            if code is None:
                continue
            for pos in positions:
                prefix = code[: pos + 1]
                edge_weight[(user_node, prefix)] += 1
                prefix_count[prefix] += 1

    prefix_nodes = {prefix: node for node, prefix in enumerate(sorted(prefix_count))}
    out_dir = args.out or (ROOT / "data" / "user_prefix_graph" / f"{args.family}_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_user_nodes: dict[int, set[int]] = defaultdict(set)

    with (out_dir / "edges.tsv").open("w", encoding="utf-8") as handle:
        handle.write("user_node\tprefix_node\tdepth\tprefix\tweight\n")
        for (user_node, prefix), weight in sorted(edge_weight.items()):
            prefix_node = prefix_nodes[prefix]
            handle.write(
                f"{user_node}\t{prefix_node}\t{len(prefix)}\t{json.dumps(prefix)}\t{weight}\n"
            )
            per_user_nodes[user_node].add(prefix_node)
    with (out_dir / "train.txt").open("w", encoding="utf-8") as handle:
        for user_node in range(len(user_nodes)):
            nodes = sorted(per_user_nodes.get(user_node, ()))
            handle.write(" ".join(map(str, [user_node, *nodes])) + "\n")
    # SASRec requires order.  Preserve train-item chronology and expand each purchase into
    # root-to-leaf prefix events; do not use LightGCN's unordered set-style train.txt for it.
    sasrec_tokens = 0
    with (out_dir / "sasrec_train.txt").open("w", encoding="utf-8") as handle:
        for user in sorted(histories):
            sequence: list[int] = []
            for item in histories[user]:
                code = item_codes.get(item)
                if code is None:
                    continue
                for pos in positions:
                    sequence.append(prefix_nodes[code[: pos + 1]])
            sasrec_tokens += len(sequence)
            handle.write(" ".join(map(str, [user_nodes[user], *sequence])) + "\n")
    with (out_dir / "user_id_map.tsv").open("w", encoding="utf-8") as handle:
        handle.write("user_node\tremapped_user_id\n")
        for user, user_node in sorted(user_nodes.items(), key=lambda pair: pair[1]):
            handle.write(f"{user_node}\t{user}\n")
    with (out_dir / "prefix_id_map.tsv").open("w", encoding="utf-8") as handle:
        handle.write("prefix_node\tdepth\tprefix\n")
        for prefix, prefix_node in sorted(prefix_nodes.items(), key=lambda pair: pair[1]):
            handle.write(f"{prefix_node}\t{len(prefix)}\t{json.dumps(prefix)}\n")

    depth_counts = Counter(len(prefix) for prefix in prefix_nodes)
    n_users, n_prefixes, n_edges = len(user_nodes), len(prefix_nodes), len(edge_weight)
    meta = {
        "graph_type": "user_exact_sid_prefix_bipartite",
        "tag": args.tag,
        "family": args.family,
        "category": category,
        # Callers may pass repository-relative or absolute paths.
        "sid_file": _display_path(sid_path),
        "sid_width": width,
        "positions_used": positions,
        "holdout": {"valid": args.valid_holdout, "test": args.test_holdout},
        "n_user_nodes": n_users,
        "n_prefix_nodes": n_prefixes,
        "prefix_nodes_by_depth": {str(depth): count for depth, count in sorted(depth_counts.items())},
        "n_unique_edges": n_edges,
        "density": n_edges / max(n_users * n_prefixes, 1),
        "avg_prefixes_per_user": sum(map(len, per_user_nodes.values())) / max(n_users, 1),
        "n_sasrec_prefix_events": sasrec_tokens,
        "node_contract": "LightGCN users are user_node; items are prefix_node; both are contiguous 0-indexed.",
        "sasrec_contract": (
            "sasrec_train.txt preserves train-item chronology and expands each purchase root-to-leaf; "
            "it is an explicit sequence-model ablation, while train.txt is the unordered LightGCN graph input."
        ),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(
        f"[prefix-graph/{args.family}/{args.tag}] users={n_users} prefixes={n_prefixes} "
        f"edges={n_edges} density={meta['density']:.6f} depths={dict(sorted(depth_counts.items()))} "
        f"-> {_display_path(out_dir)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
