#!/usr/bin/env python
"""Fixed-depth feasibility audit for C3 prefix handoff.

At each SID depth, decode only that many tokens, expand every live beam prefix to
all exact catalog items below it, then rank candidates with

    score(item) = decoder_prefix_score + alpha * cosine(user, item_leaf_prefix)

The TIGER checkpoint is frozen.  This script is an audit, not the final adaptive
stopping controller: it establishes whether behavioral reranking can recover (or
improve) recommendation quality before the identity-heavy suffix is decoded.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from recbole3.config import instantiate_dataclass
from recbole3.dataset import ITEM_ID, SEEN_ITEM_IDS, USER_ID, get_dataset_spec
from recbole3.model import get_model_spec
from recbole3.run import compose_config
from recbole3.utils import require_component_cfg, require_component_name


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--emb-dir", default="", help="Frozen user/prefix embeddings; omit for decoder-only alpha=0 diagnostics.")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--alphas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--depths", default="", help="Comma-separated SID depths; empty means all.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--trace", action="store_true", help="Save per-user predictions and causal stopping features.")
    parser.add_argument("--partial-trace", action="store_true", help="Allow a diagnostic trace without full-depth native fallback.")
    parser.add_argument("--candidate-trace", action="store_true", help="Also export frozen candidate scores for validation-only reranker selection; requires --trace.")
    args, overrides = parser.parse_known_args()
    if args.candidate_trace and not args.trace:
        parser.error("--candidate-trace requires --trace")
    if args.partial_trace and (not args.trace or args.candidate_trace):
        parser.error("--partial-trace requires --trace without --candidate-trace")
    if not args.emb_dir and (args.alphas.strip() not in {"0", "0.0"}):
        parser.error("Omitting --emb-dir supports only decoder-only --alphas 0 without --candidate-trace")
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    return args, overrides


def _records_list(records: Any) -> list[Mapping[str, Any]]:
    if hasattr(records, "to_dict"):
        return records.to_dict("records")
    return list(records)


def _load_maps(emb_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[tuple[int, ...], int]]:
    user_emb = np.load(emb_dir / "user_emb.npy").astype(np.float32)
    prefix_emb = np.load(emb_dir / "item_emb.npy").astype(np.float32)
    user_emb /= np.maximum(np.linalg.norm(user_emb, axis=1, keepdims=True), 1e-12)
    prefix_emb /= np.maximum(np.linalg.norm(prefix_emb, axis=1, keepdims=True), 1e-12)

    user_rows: dict[int, int] = {}
    with (emb_dir.parent / "user_id_map.tsv").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            node, user_id = line.rstrip("\n").split("\t")
            user_rows[int(user_id)] = int(node)

    prefix_rows: dict[tuple[int, ...], int] = {}
    with (emb_dir.parent / "prefix_id_map.tsv").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            node, _depth, raw_prefix = line.rstrip("\n").split("\t")
            prefix_rows[tuple(int(v) for v in json.loads(raw_prefix))] = int(node)

    if user_emb.shape[0] != len(user_rows) or prefix_emb.shape[0] != len(prefix_rows):
        raise ValueError("Embedding rows do not match graph maps.")
    return user_emb, prefix_emb, user_rows, prefix_rows


def _metric_state(alphas: Sequence[float], depths: Sequence[int]) -> dict[str, dict[str, float]]:
    keys = ["native_full"] + [f"d{d}_a{a:g}" for d in depths for a in alphas]
    return {key: {"n": 0.0, "hit5": 0.0, "hit10": 0.0, "ndcg5": 0.0, "ndcg10": 0.0} for key in keys}


def _add_metric(state: dict[str, float], ranking: Sequence[int], target: int) -> None:
    state["n"] += 1
    try:
        rank = list(ranking[:10]).index(target)
    except ValueError:
        return
    gain = 1.0 / math.log2(rank + 2)
    state["hit10"] += 1
    state["ndcg10"] += gain
    if rank < 5:
        state["hit5"] += 1
        state["ndcg5"] += gain


def _finalize(state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, row in state.items():
        n = max(row["n"], 1.0)
        out[key] = {
            "n_users": int(row["n"]),
            "recall@5": row["hit5"] / n,
            "ndcg@5": row["ndcg5"] / n,
            "recall@10": row["hit10"] / n,
            "ndcg@10": row["ndcg10"] / n,
        }
    return out


def _excluded(record: Mapping[str, Any]) -> set[int]:
    values = record.get(SEEN_ITEM_IDS)
    return {int(v) for v in (() if values is None else values)}


def _fill(ranking: list[int], excluded: set[int], fallback: Sequence[int], k: int = 10) -> int:
    if len(ranking) >= k:
        del ranking[k:]
        return 0
    before = len(ranking)
    used = set(ranking)
    for item in fallback:
        item = int(item)
        if item in used or item in excluded:
            continue
        ranking.append(item)
        used.add(item)
        if len(ranking) >= k:
            break
    return len(ranking) - before


def _candidate_order(
    candidate_ids: np.ndarray,
    decoder_scores: np.ndarray,
    behavior_scores: np.ndarray,
    beam_orders: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Rank candidates while preserving native beam order on exact score ties.

    At full SID depth, native TIGER decoding consumes returned beams in their
    score order.  Items tied exactly on decoder score must therefore retain the
    first compatible beam's order, rather than receive an arbitrary item-ID
    tie break.  This also gives shallow handoff rankings a deterministic,
    decoder-consistent tie rule.
    """
    combined = decoder_scores + float(alpha) * behavior_scores
    return np.lexsort((candidate_ids, beam_orders, -decoder_scores, -combined))


def main() -> None:
    args, overrides = _parse_args()
    alphas = tuple(float(v) for v in args.alphas.split(",") if v.strip())
    cfg = compose_config(overrides=overrides)
    runtime_cfg = require_component_cfg(cfg, "runtime")
    dataset_cfg = require_component_cfg(cfg, "dataset")
    model_cfg = require_component_cfg(cfg, "model")
    trainer_cfg = require_component_cfg(cfg, "trainer")

    model_spec = get_model_spec(require_component_name(model_cfg, "model"))
    dataset_spec = get_dataset_spec(require_component_name(dataset_cfg, "dataset"))
    dataset = dataset_spec.dataset_cls(instantiate_dataclass(dataset_spec.config_cls, dataset_cfg))
    model = model_spec.model_cls(instantiate_dataclass(model_spec.config_cls, model_cfg))
    trainer = model_spec.trainer_cls(instantiate_dataclass(model_spec.trainer_config_cls, trainer_cfg))
    trainer._runtime_cfg = runtime_cfg
    task_data = dataset.prepare(eval_config=trainer.config.eval)
    prepared = model_spec.model_data_cls.from_task_dataset(task_data, model_config=model_cfg)
    model.build_eval_collator(prepared)
    state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    codec = model._sid_codec()
    t5 = model._t5_module()
    depths = (
        tuple(int(v) for v in args.depths.split(",") if v.strip())
        if args.depths.strip()
        else tuple(range(1, int(codec.n_digit) + 1))
    )
    if not depths or any(d < 1 or d > int(codec.n_digit) for d in depths):
        raise ValueError(f"Depths must be in [1, {codec.n_digit}], got {depths}.")

    emb_dir = Path(args.emb_dir) if args.emb_dir else None
    if emb_dir is not None:
        user_emb, prefix_emb, user_rows, prefix_rows = _load_maps(emb_dir)
        item_leaf_row = {
            item: prefix_rows.get(tuple(int(v) for v in sid))
            for item, sid in codec.item_to_sid.items()
        }
        item_vectors = np.zeros((len(item_leaf_row), prefix_emb.shape[1]), dtype=np.float32)
        for item, leaf in item_leaf_row.items():
            if leaf is not None:
                item_vectors[item] = prefix_emb[leaf]
    else:
        user_emb = prefix_emb = item_vectors = None
        user_rows = prefix_rows = {}
        item_leaf_row = {item: None for item in codec.item_to_sid}
    if args.candidate_trace:
        raw_users = np.load(emb_dir / "user_emb.npy").astype(np.float32) if emb_dir else np.zeros((1, 1), np.float32)
        raw_prefixes = np.load(emb_dir / "item_emb.npy").astype(np.float32) if emb_dir else np.zeros((1, 1), np.float32)
        raw_leaves = np.zeros((len(item_leaf_row), 1), np.float32) if item_vectors is None else np.zeros_like(item_vectors)
        raw_roots = np.zeros((len(item_leaf_row), 1), np.float32) if item_vectors is None else np.zeros_like(item_vectors)
        for item, sid in codec.item_to_sid.items():
            leaf = item_leaf_row[item]
            root = prefix_rows.get(tuple(int(v) for v in sid[:1]))
            if leaf is not None:
                raw_leaves[item] = raw_prefixes[leaf]
            if root is not None:
                raw_roots[item] = raw_prefixes[root]
        candidate_trace = defaultdict(list)
        candidate_offsets = [0]
    prefix_buckets: dict[int, dict[tuple[int, ...], tuple[int, ...]]] = {}
    for depth in depths:
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for item, tokens in codec.item_to_tokens.items():
            buckets[tuple(tokens[:depth])].append(int(item))
        prefix_buckets[depth] = {prefix_: tuple(items) for prefix_, items in buckets.items()}

    method = trainer.create_evaluation_method(prepared)
    collate = method.build_eval_collate_fn(model, prepared)
    eval_dataset = prepared.get_eval_dataset(args.split)
    dataloader = trainer.build_dataloader(eval_dataset, collate, shuffle=False)
    metrics = _metric_state(alphas, depths)
    stats = {
        str(d): {"queries": 0, "candidate_sum": 0, "target_covered": 0, "fallback_items": 0, "seconds": 0.0}
        for d in depths
    }
    processed = 0
    traces = defaultdict(list)

    with torch.inference_mode():
        for model_inputs, raw_records in dataloader:
            records = _records_list(raw_records)
            if args.max_users and processed >= args.max_users:
                break
            if args.max_users and processed + len(records) > args.max_users:
                keep = args.max_users - processed
                records = records[:keep]
                model_inputs = {k: v[:keep] for k, v in model_inputs.items()}
            batch_size = len(records)
            batch_trace = {
                "predictions": np.empty((batch_size, len(depths), len(alphas), 10), dtype=np.int32),
                "entropy": np.zeros((batch_size, len(depths)), dtype=np.float32),
                "candidates": np.zeros((batch_size, len(depths)), dtype=np.int32),
                "target_covered": np.zeros((batch_size, len(depths)), dtype=bool),
                "behavior_margin": np.full((batch_size, len(depths)), -1.0, dtype=np.float32),
                "embedding_complete": np.zeros((batch_size, len(depths)), dtype=bool),
                "rank_margin": np.full((batch_size, len(depths), len(alphas)), -1.0, dtype=np.float32),
                "native": np.full((batch_size, 10), -1, dtype=np.int32),
            } if args.trace else None
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)
            user_ids_tensor = model_inputs.get("user_ids")
            if user_ids_tensor is not None:
                user_ids_tensor = user_ids_tensor.to(device)
            encoder_kwargs = model._encoder_kwargs(input_ids, attention_mask, user_ids_tensor, None, None)
            if emb_dir is not None:
                batch_profiles = np.stack([
                    user_emb[user_rows[int(r[USER_ID])]] if int(r[USER_ID]) in user_rows
                    else np.zeros(user_emb.shape[1], dtype=np.float32) for r in records
                ])
                batch_cosines = batch_profiles @ item_vectors.T
            else:
                batch_cosines = None
            if args.candidate_trace:
                raw_profiles = np.stack([
                    raw_users[user_rows[int(r[USER_ID])]] if int(r[USER_ID]) in user_rows
                    else np.zeros(raw_users.shape[1], dtype=np.float32) for r in records
                ])
                batch_dots = raw_profiles @ raw_leaves.T
                batch_root_dots = raw_profiles @ raw_roots.T

            for depth_index, depth in enumerate(depths):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                generation_length = depth + 1 if depth == codec.n_digit else depth
                generation_kwargs = {
                    "max_new_tokens": generation_length,
                    "num_beams": int(model.config.num_beams),
                    "num_return_sequences": int(model.config.num_beams),
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "use_cache": True,
                }
                # Shallow audits need exactly d code tokens.  Full decoding must retain
                # TIGER's native EOS behavior; forcing min_new_tokens changes beam order.
                if depth < codec.n_digit:
                    generation_kwargs["min_new_tokens"] = generation_length
                generated = t5.generate(
                    **generation_kwargs,
                    **encoder_kwargs,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                stats[str(depth)]["seconds"] += time.perf_counter() - started
                beam_width = int(model.config.num_beams)
                sequences = generated.sequences.reshape(batch_size, beam_width, -1)
                seq_scores = generated.sequences_scores.reshape(batch_size, beam_width).detach().cpu().numpy()
                prefixes = sequences[:, :, 1 : 1 + depth].detach().cpu().numpy()
                if args.trace:
                    # Entropy of the distribution that emitted the current token. Already
                    # available at depth d; never inspect the next-token forward pass.
                    scores = generated.scores[min(depth, len(generated.scores)) - 1].float()
                    probs = scores.softmax(-1)
                    ent = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
                    support = torch.isfinite(scores).sum(-1).clamp_min(2).float().log()
                    batch_trace["entropy"][:, depth_index] = (ent / support).reshape(batch_size, -1).mean(-1).cpu().numpy()

                for row, record in enumerate(records):
                    target = int(record[ITEM_ID])
                    user_id = int(record[USER_ID])
                    excluded = _excluded(record)
                    # Item -> (best compatible beam score, earliest such beam).
                    # Keeping the beam position makes alpha=0 full-depth handoff
                    # exactly reproduce native beam ordering under score ties.
                    decoder_by_item: dict[int, tuple[float, int]] = {}
                    for beam in range(beam_width):
                        prefix = tuple(int(v) for v in prefixes[row, beam].tolist())
                        decoder_score = float(seq_scores[row, beam])
                        for item in prefix_buckets[depth].get(prefix, ()):
                            if item in excluded:
                                continue
                            previous = decoder_by_item.get(item)
                            if previous is None or decoder_score > previous[0]:
                                decoder_by_item[item] = (decoder_score, beam)

                    candidates = list(decoder_by_item)
                    stats[str(depth)]["queries"] += 1
                    stats[str(depth)]["candidate_sum"] += len(candidates)
                    stats[str(depth)]["target_covered"] += int(target in decoder_by_item)
                    if args.trace:
                        batch_trace["target_covered"][row, depth_index] = target in decoder_by_item
                    user_row = user_rows.get(user_id) if emb_dir is not None else None
                    behavior: dict[int, float] = {}
                    if emb_dir is not None and user_row is not None:
                        behavior = {item: float(batch_cosines[row, item]) for item in candidates}
                    else:
                        behavior = {item: 0.0 for item in candidates}

                    if args.trace:
                        batch_trace["candidates"][row, depth_index] = len(candidates)
                        batch_trace["embedding_complete"][row, depth_index] = (
                            user_row is not None and bool(candidates)
                            and all(item_leaf_row.get(item) is not None for item in candidates)
                        )
                        beh_sorted = sorted(behavior.values(), reverse=True)
                        if len(beh_sorted) > 10:
                            batch_trace["behavior_margin"][row, depth_index] = beh_sorted[9] - beh_sorted[10]

                    cand_ids = np.asarray(candidates, dtype=np.int32)
                    dec_values = np.asarray([decoder_by_item[i][0] for i in candidates], dtype=np.float64)
                    beam_values = np.asarray([decoder_by_item[i][1] for i in candidates], dtype=np.int32)
                    beh_values = np.asarray([behavior[i] for i in candidates], dtype=np.float64)
                    if args.candidate_trace:
                        fallback: list[int] = []
                        _fill(fallback, excluded, codec.fallback_item_ids)
                        candidate_trace["query_index"].append(processed + row)
                        candidate_trace["depth"].append(depth)
                        candidate_trace["fallback"].append(fallback)
                        for key, value in (
                            ("items", cand_ids), ("decoder", dec_values),
                            ("beam", beam_values), ("cosine", beh_values),
                            ("dot", batch_dots[row, cand_ids]),
                            ("root_dot", batch_root_dots[row, cand_ids]),
                            ("covered", np.asarray([user_row is not None and item_leaf_row[i] is not None for i in candidates], dtype=bool)),
                        ):
                            candidate_trace[key].append(value)
                        candidate_offsets.append(candidate_offsets[-1] + len(candidates))
                    for alpha_index, alpha in enumerate(alphas):
                        order = _candidate_order(cand_ids, dec_values, beh_values, beam_values, alpha)[:11]
                        ranking = cand_ids[order].tolist()
                        if args.trace and len(ranking) > 10:
                            top, nxt = ranking[9:11]
                            batch_trace["rank_margin"][row, depth_index, alpha_index] = (
                                decoder_by_item[top][0] + alpha * behavior[top]
                                - decoder_by_item[nxt][0] - alpha * behavior[nxt]
                            )
                        ranking = ranking[:10]
                        added = _fill(ranking, excluded, codec.fallback_item_ids)
                        stats[str(depth)]["fallback_items"] += added
                        _add_metric(metrics[f"d{depth}_a{alpha:g}"], ranking, target)
                        if args.trace:
                            batch_trace["predictions"][row, depth_index, alpha_index] = ranking

                    if depth == codec.n_digit:
                        native: list[int] = []
                        used: set[int] = set()
                        for beam in range(beam_width):
                            item = codec.token_tuple_to_item(prefixes[row, beam].tolist())
                            if item is None or item in excluded or item in used:
                                continue
                            native.append(item)
                            used.add(item)
                            if len(native) == 10:
                                break
                        _fill(native, excluded, codec.fallback_item_ids)
                        _add_metric(metrics["native_full"], native, target)
                        if args.trace:
                            batch_trace["native"][row] = native

            if args.trace:
                for key, value in batch_trace.items():
                    traces[key].append(value)
                traces["user_ids"].append(np.asarray([int(r[USER_ID]) for r in records]))
                traces["targets"].append(np.asarray([int(r[ITEM_ID]) for r in records]))

            processed += batch_size
            print(f"[handoff] split={args.split} users={processed}", flush=True)

    for depth in depths:
        row = stats[str(depth)]
        q = max(int(row["queries"]), 1)
        row["mean_candidates"] = row.pop("candidate_sum") / q
        row["target_coverage"] = row.pop("target_covered") / q
        row["mean_fallback_items_per_alpha"] = row.pop("fallback_items") / (q * max(len(alphas), 1))

    payload = {
        "kind": "fixed_depth_prefix_handoff_feasibility",
        "split": args.split,
        "checkpoint": args.ckpt,
        "embedding_dir": args.emb_dir,
        "formula": "decoder_sequence_score + alpha * cosine(user_embedding, full_leaf_prefix_embedding)",
        "alphas": list(alphas),
        "sid_depth": int(codec.n_digit),
        "beam_width": int(model.config.num_beams),
        "leaf_embedding_coverage": {
            "covered": sum(v is not None for v in item_leaf_row.values()) if emb_dir is not None else 0,
            "catalog": len(item_leaf_row),
        },
        "metrics": _finalize(metrics),
        "depth_stats": stats,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.trace:
        if codec.n_digit not in depths and not args.partial_trace:
            raise ValueError("Traces require full depth for a verified native fallback.")
        trace_path = output.with_suffix(".npz")
        np.savez_compressed(trace_path, **{k: np.concatenate(v) for k, v in traces.items()},
                            depths=np.asarray(depths), alphas=np.asarray(alphas))
        payload["trace_file"] = str(trace_path)
        payload["trace_includes_native_full"] = codec.n_digit in depths
        payload["entropy_definition"] = "mean normalized entropy over live parent beams for the token just emitted; no next-step peek"
    if args.candidate_trace:
        candidate_path = output.with_suffix(".candidates.npz")
        row_keys = {"query_index", "depth", "fallback"}
        np.savez_compressed(
            candidate_path,
            **{key: np.asarray(value) if key in row_keys else np.concatenate(value)
               for key, value in candidate_trace.items()},
            offsets=np.asarray(candidate_offsets, dtype=np.int64),
            user_ids=np.concatenate(traces["user_ids"]),
            targets=np.concatenate(traces["targets"]),
            native=np.concatenate(traces["native"]),
            sid_depth=np.asarray(codec.n_digit), split=np.asarray(args.split),
        )
        payload["candidate_trace_file"] = str(candidate_path)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
