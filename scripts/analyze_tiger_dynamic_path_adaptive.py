#!/usr/bin/env python3
"""Validation-select a per-user one-token-early dynamic-path CF handoff.

The frozen decoder and SASRec produce rankings at depth L-1 and L.  A gate
observes only depth-(L-1) quantities and decides whether each user stops or
continues.  Validation chooses a scalar threshold near a requested nominal
step-saving band; test is evaluated once with that frozen rule.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.std())
    return (values - values.mean()) / scale if scale > 1e-8 else np.zeros_like(values)


def complete(items, order, fallback):
    # Keep the full ordered candidate list until fallback de-duplication is
    # complete. A fallback item can duplicate an early candidate; retaining
    # lower-ranked candidates guarantees a valid Top-10 whenever the combined
    # pool contains ten distinct items.
    selected = [int(value) for value in items[order]]
    used = set(selected)
    for value in fallback:
        value = int(value)
        if value not in used:
            selected.append(value)
            used.add(value)
        if len(selected) >= 10:
            break
    if len(selected) < 10:
        raise ValueError("Candidate and fallback pools cannot fill Top-10")
    return selected[:10]


def contributions(predictions, targets):
    hit = predictions == targets[:, None]
    discount = 1 / np.log2(np.arange(10) + 2)
    return np.column_stack((hit.any(1), (hit * discount).sum(1))).astype(np.float64)


def metrics(values, baseline):
    mean = values.mean(0)
    base = baseline.mean(0)
    return {
        "recall@10": float(mean[0]),
        "ndcg@10": float(mean[1]),
        "recall_change_pct": float(100 * (mean[0] / base[0] - 1)),
        "ndcg_change_pct": float(100 * (mean[1] / base[1] - 1)),
    }


def rankings_and_features(data, alpha):
    count = len(data["targets"])
    predictions = np.empty_like(data["native"])
    decoder_predictions = np.empty_like(data["native"])
    features = np.zeros((count, 6), dtype=np.float64)
    for query in range(count):
        lo, hi = int(data["offsets"][query]), int(data["offsets"][query + 1])
        items = data["items"][lo:hi]
        decoder = data["decoder"][lo:hi].astype(np.float64)
        beam = data["beam"][lo:hi]
        cf = data["cf_logp"][lo:hi].astype(np.float64)
        decoder_z = standardize(decoder)
        cf_z = np.column_stack([standardize(cf[:, depth]) for depth in range(cf.shape[1])])
        cf_score = (cf_z * alpha).sum(1) / math.sqrt(len(alpha))
        fused = decoder_z + cf_score
        fused_order = np.lexsort((items, beam, -decoder, -fused))
        decoder_order = np.lexsort((items, beam, -decoder))
        predictions[query] = complete(items, fused_order, data["fallback"][query])
        decoder_predictions[query] = complete(items, decoder_order, data["fallback"][query])

        def boundary_margin(values, order):
            return float(values[order[9]] - values[order[10]]) if len(order) > 10 else np.inf

        overlap = len(set(predictions[query]).intersection(decoder_predictions[query]))
        union = 20 - overlap
        features[query] = (
            math.log1p(len(items)),
            boundary_margin(decoder_z, decoder_order),
            boundary_margin(fused, fused_order),
            boundary_margin(cf_score, np.argsort(-cf_score, kind="stable")),
            1 - overlap / union if union else 0.0,
            float(fused[fused_order[0]] - fused[fused_order[1]]) if len(fused_order) > 1 else np.inf,
        )
    return predictions, decoder_predictions, features


def entropy_column(base_path: Path, depth: int):
    base = load_npz(base_path)
    depths = base["depths"].astype(int)
    where = np.flatnonzero(depths == depth)
    if len(where) != 1:
        raise ValueError(f"Depth {depth} missing from {base_path}")
    return base["entropy"][:, int(where[0])].astype(np.float64)


def candidate_rules(
    features, names, minimum_saving, maximum_saving, per_exit_saving,
    grid_size=17, include_never=False,
):
    lower_fraction = minimum_saving / (100 * per_exit_saving)
    upper_fraction = min(1.0, maximum_saving / (100 * per_exit_saving))
    quantiles = np.unique(np.r_[np.linspace(lower_fraction, upper_fraction, grid_size), upper_fraction])
    rules = []
    if include_never:
        rules.append({
            "feature": "never_exit", "column": -1, "direction": "none",
            "threshold": None, "validation_exit_fraction": 0.0,
            "validation_saving_pct": 0.0,
            "exits": np.zeros(len(features), dtype=bool),
        })
    for column, name in enumerate(names):
        values = features[:, column]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        clean = values[finite]
        for direction in ("low", "high"):
            for fraction in quantiles:
                if direction == "low":
                    threshold = float(np.quantile(clean, fraction))
                    exits = values <= threshold
                else:
                    threshold = float(np.quantile(clean, 1 - fraction))
                    exits = values >= threshold
                saving = 100 * per_exit_saving * exits.mean()
                if minimum_saving - 1e-9 <= saving <= maximum_saving + 1e-9:
                    rules.append({
                        "feature": name, "column": column, "direction": direction,
                        "threshold": threshold, "validation_exit_fraction": float(exits.mean()),
                        "validation_saving_pct": float(saving), "exits": exits,
                    })
    return rules


def apply_rule(features, rule):
    if rule["direction"] == "none":
        return np.zeros(len(features), dtype=bool)
    values = features[:, int(rule["column"])]
    return values <= rule["threshold"] if rule["direction"] == "low" else values >= rule["threshold"]


def evaluate_split(early_values, full_values, native_values, exits, per_exit_saving):
    values = full_values.copy()
    values[exits] = early_values[exits]
    result = metrics(values, native_values)
    result.update({
        "n": int(len(exits)), "early_exits": int(exits.sum()),
        "full_decodes": int((~exits).sum()),
        "nominal_step_reduction_pct": float(100 * per_exit_saving * exits.mean()),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--valid-base", type=Path, required=True)
    parser.add_argument("--test-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-saving-pct", type=float, default=18.0)
    parser.add_argument("--max-saving-pct", type=float, default=20.0)
    parser.add_argument(
        "--max-degradation-pct", type=float, default=None,
        help="If set, maximize validation saving while both metric changes stay above -VALUE.",
    )
    parser.add_argument("--tolerance-grid-size", type=int, default=201)
    args = parser.parse_args()

    early_valid = load_npz(args.pair_dir / "early" / "valid.npz")
    full_valid = load_npz(args.pair_dir / "full" / "valid.npz")
    early_test = load_npz(args.pair_dir / "early" / "test.npz")
    full_test = load_npz(args.pair_dir / "full" / "test.npz")
    early_depth = int(early_valid["candidate_depth"])
    width = int(early_valid["cf_logp"].shape[1])
    if early_depth != width - 1:
        raise ValueError(f"Expected one-token-early depth {width - 1}, got {early_depth}")
    per_exit_saving = 1 / width

    def alpha(directory):
        return np.asarray(json.loads((directory / "result.json").read_text())["learned_alpha_by_depth"])

    ev_pred, _, ev_features = rankings_and_features(early_valid, alpha(args.pair_dir / "early"))
    fv_pred, _, _ = rankings_and_features(full_valid, alpha(args.pair_dir / "full"))
    et_pred, _, et_features = rankings_and_features(early_test, alpha(args.pair_dir / "early"))
    ft_pred, _, _ = rankings_and_features(full_test, alpha(args.pair_dir / "full"))
    ev_features = np.column_stack((ev_features, entropy_column(args.valid_base, early_depth)))
    et_features = np.column_stack((et_features, entropy_column(args.test_base, early_depth)))
    names = [
        "log_candidate_count", "decoder_top10_margin", "fused_top10_margin",
        "cf_top10_margin", "decoder_fused_top10_jaccard_distance",
        "fused_top1_margin", "decoder_entropy",
    ]

    ev_native = contributions(early_valid["native"], early_valid["targets"])
    et_native = contributions(early_test["native"], early_test["targets"])
    ev_early = contributions(ev_pred, early_valid["targets"])
    ev_full = contributions(fv_pred, full_valid["targets"])
    et_early = contributions(et_pred, early_test["targets"])
    et_full = contributions(ft_pred, full_test["targets"])

    tolerance_mode = args.max_degradation_pct is not None
    if tolerance_mode and args.max_degradation_pct < 0:
        raise ValueError("--max-degradation-pct must be nonnegative")
    minimum_saving = 0.0 if tolerance_mode else args.min_saving_pct
    maximum_saving = 100 * per_exit_saving if tolerance_mode else args.max_saving_pct
    selected = None
    for rule in candidate_rules(
        ev_features, names, minimum_saving, maximum_saving, per_exit_saving,
        grid_size=args.tolerance_grid_size if tolerance_mode else 17,
        include_never=tolerance_mode,
    ):
        report = evaluate_split(ev_early, ev_full, ev_native, rule.pop("exits"), per_exit_saving)
        if tolerance_mode:
            if min(report["recall_change_pct"], report["ndcg_change_pct"]) < -args.max_degradation_pct:
                continue
            # Performance is a constraint; among feasible policies, save the
            # most work, then prefer the strongest metric margin.
            key = (
                report["nominal_step_reduction_pct"],
                min(report["recall_change_pct"], report["ndcg_change_pct"]),
                report["recall_change_pct"] + report["ndcg_change_pct"],
            )
        else:
            # Prefer policies improving both metrics; then maximize the weaker gain,
            # total gain, and closeness to the requested upper saving target.
            key = (
                min(report["recall_change_pct"], report["ndcg_change_pct"]),
                report["recall_change_pct"] + report["ndcg_change_pct"],
                -abs(report["nominal_step_reduction_pct"] - args.max_saving_pct),
            )
        if selected is None or key > selected[0]:
            selected = (key, rule, report)
    if selected is None:
        raise ValueError("No validation gate satisfies the selection constraint")

    _, rule, validation = selected
    test_exits = apply_rule(et_features, rule)
    test = evaluate_split(et_early, et_full, et_native, test_exits, per_exit_saving)
    payload = {
        "kind": "adaptive_one_token_early_dynamic_path_cf",
        "selection": (
            "Validation-only scalar observable gate; maximize nominal saving while both Recall/NDCG changes remain within the requested degradation tolerance."
            if tolerance_mode else
            "Validation-only scalar observable gate; maximize the weaker Recall/NDCG gain inside the requested saving band."
        ),
        "early_depth": early_depth, "full_depth": width,
        "requested_saving_band_pct": None if tolerance_mode else [args.min_saving_pct, args.max_saving_pct],
        "max_validation_degradation_pct": args.max_degradation_pct,
        "gate": rule, "validation": validation, "test": test,
        "full_pathfusion_validation": metrics(ev_full, ev_native),
        "full_pathfusion_test": metrics(et_full, et_native),
        "fixed_early_validation": metrics(ev_early, ev_native),
        "fixed_early_test": metrics(et_early, et_native),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
