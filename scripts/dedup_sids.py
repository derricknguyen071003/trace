"""Make semantic-ID tuples globally unique for TIGER.

TIGERSIDCodec requires globally unique tuples. This appends a per-tuple collision
counter as an extra token (first occurrence -> 0, next -> 1, ...), so
[6, 27, 220, 64] becomes [6, 27, 220, 64, 0]. Items are processed in item-ID
order for determinism. ``--only-if-collisions`` preserves the original tuple
width when the input is already injective. ``--disjoint-suffix`` places the
counter above every source token value so the identity position cannot share a
model embedding with a semantic code. Use both options for matched C1.

    python scripts/dedup_sids.py <input item_sids.json> <output item_sids_dedup.json>
    python scripts/dedup_sids.py --only-if-collisions <input item_sids.json> <output item_sids_dedup.json>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-if-collisions",
        action="store_true",
        help="preserve width when all input tuples are already unique",
    )
    parser.add_argument(
        "--disjoint-suffix",
        action="store_true",
        help="offset collision counters above the maximum input token value",
    )
    parser.add_argument("input", help="input item-SID JSON")
    parser.add_argument("output", help="output item-SID JSON")
    args = parser.parse_args()
    in_path, out_path = args.input, args.output

    with open(in_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    rows = [(item_id, tuple(int(x) for x in raw[item_id])) for item_id in sorted(raw, key=int)]
    counts: Counter[tuple[int, ...]] = Counter(sid for _, sid in rows)
    if args.only_if_collisions and all(count == 1 for count in counts.values()):
        dedup = {item_id: list(sid) for item_id, sid in rows}
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(dedup, handle)
        print(
            f"[dedup_sids] {len(dedup)} items -> {out_path} | "
            "colliding tuples=0 | preserved original width"
        )
        return

    suffix_base = (
        max((value for _, sid in rows for value in sid), default=-1) + 1
        if args.disjoint_suffix
        else 0
    )
    seen: Counter[tuple[int, ...]] = Counter()
    dedup: dict[str, list[int]] = {}
    for item_id in sorted(raw, key=int):
        sid = tuple(int(x) for x in raw[item_id])
        count = seen[sid]
        seen[sid] += 1
        dedup[item_id] = list(sid) + [suffix_base + count]

    max_extra = max(seen.values()) - 1 if seen else 0
    total = len(dedup)
    collided = sum(1 for v in seen.values() if v > 1)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(dedup, handle)
    print(
        f"[dedup_sids] {total} items -> {out_path} | "
        f"colliding tuples={collided} | max dedup counter={max_extra} | "
        f"suffix base={suffix_base}"
    )


if __name__ == "__main__":
    main()
