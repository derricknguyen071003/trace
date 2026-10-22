#!/usr/bin/env python3
"""Learn one bounded CF fusion weight per SID depth with complete-item listwise loss."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def load(path: Path):
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def padded(data):
    offsets, decoder, cf = data["offsets"], data["decoder"], data["cf_logp"]
    count = len(offsets) - 1
    width = cf.shape[1]
    maximum = int(np.diff(offsets).max())
    values = np.full((count, maximum, width + 1), 0.0, dtype=np.float32)
    mask = np.zeros((count, maximum), dtype=bool)
    target_index = np.full(count, -1, dtype=np.int64)
    for query in range(count):
        lo, hi = int(offsets[query]), int(offsets[query + 1])
        n = hi - lo
        features = np.column_stack((decoder[lo:hi], cf[lo:hi]))
        mean = features.mean(0, keepdims=True)
        std = features.std(0, keepdims=True)
        values[query, :n] = (features - mean) / np.maximum(std, 1e-6)
        mask[query, :n] = True
        match = np.flatnonzero(data["items"][lo:hi] == data["targets"][query])
        if len(match):
            target_index[query] = int(match[0])
    return values, mask, target_index


def scores(features: torch.Tensor, raw_alpha: torch.Tensor):
    alpha = raw_alpha.sigmoid()
    return features[..., 0] + (features[..., 1:] * alpha).sum(-1) / math.sqrt(len(alpha))


def ranking_metrics(data, raw_alpha):
    features, _, _ = padded(data)
    alpha = torch.as_tensor(raw_alpha, dtype=torch.float32)
    score = scores(torch.as_tensor(features), alpha).numpy()
    predictions = np.empty_like(data["native"])
    for query in range(len(predictions)):
        lo, hi = int(data["offsets"][query]), int(data["offsets"][query + 1])
        n = hi - lo
        order = np.lexsort((data["items"][lo:hi], data["beam"][lo:hi],
                            -data["decoder"][lo:hi], -score[query, :n]))
        selected = data["items"][lo:hi][order].tolist()
        used = set(selected)
        for item in data["fallback"][query]:
            item = int(item)
            if item not in used:
                selected.append(item)
                used.add(item)
            if len(selected) >= 10:
                break
        predictions[query] = selected[:10]
    target = data["targets"][:, None]
    hit = predictions == target
    native_hit = data["native"] == target
    recall = hit.any(1)
    native_recall = native_hit.any(1)
    discounts = 1 / np.log2(np.arange(10) + 2)
    ndcg = (hit * discounts).sum(1)
    native_ndcg = (native_hit * discounts).sum(1)
    recall_change = float(100 * (recall.mean() / native_recall.mean() - 1)) if native_recall.any() else None
    ndcg_change = float(100 * (ndcg.mean() / native_ndcg.mean() - 1)) if native_ndcg.any() else None
    return {
        "n": len(recall), "native_recall@10": float(native_recall.mean()),
        "fused_recall@10": float(recall.mean()), "native_ndcg@10": float(native_ndcg.mean()),
        "fused_ndcg@10": float(ndcg.mean()),
        "recall_change_pct": recall_change, "ndcg_change_pct": ndcg_change,
        "rescues": int(((~native_recall) & recall).sum()),
        "harms": int((native_recall & (~recall)).sum()),
        "top10_changed": int(np.any(predictions != data["native"], axis=1).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal-step-reduction-pct", type=float, default=0.0)
    args = parser.parse_args()
    valid, test = load(args.valid), load(args.test)
    features, mask, target = padded(valid)
    keep = target >= 0
    if not keep.any():
        raise ValueError("No validation target occurs in the full-depth candidate pools")
    x = torch.as_tensor(features[keep])
    m = torch.as_tensor(mask[keep])
    y = torch.as_tensor(target[keep])
    raw_alpha = torch.nn.Parameter(torch.full((valid["cf_logp"].shape[1],), -4.0))
    optimizer = torch.optim.LBFGS([raw_alpha], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = scores(x, raw_alpha).masked_fill(~m, -torch.inf)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        return loss

    initial = float(closure().detach())
    optimizer.step(closure)
    final = float(closure().detach())
    raw = raw_alpha.detach().numpy()
    payload = {
        "kind": "frozen_dynamic_cf_complete_item_listwise_fusion",
        "pair": str(valid["pair"]), "learned_alpha_by_depth": raw_alpha.sigmoid().detach().tolist(),
        "train_queries_with_target_in_pool": int(keep.sum()),
        "train_queries": int(len(keep)), "initial_listwise_loss": initial,
        "final_listwise_loss": final, "validation": ranking_metrics(valid, raw),
        "test": ranking_metrics(test, raw),
        "nominal_step_reduction_pct": args.nominal_step_reduction_pct,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


METRICS = [('HR@1', 1, False), ('HR@5', 5, False), ('NDCG@5', 5, True), ('HR@10', 10, False), ('NDCG@10', 10, True)]

def evaluate(directory):
    result = json.loads((directory / 'result.json').read_text())
    data = load(directory / 'test.npz')
    alpha = np.asarray(result['learned_alpha_by_depth'], dtype=np.float32)
    raw = np.log(alpha / (1 - alpha))
    (features, _, _) = padded(data)
    score = scores(torch.as_tensor(features), torch.as_tensor(raw)).numpy()
    predictions = np.empty_like(data['native'])
    assert predictions.shape[1] == 10
    for q in range(len(predictions)):
        (lo, hi) = map(int, data['offsets'][q:q + 2])
        order = np.lexsort((data['items'][lo:hi], data['beam'][lo:hi], -data['decoder'][lo:hi], -score[q, :hi - lo]))
        selected = data['items'][lo:hi][order].tolist()
        used = set(selected)
        for item in data['fallback'][q]:
            item = int(item)
            if item not in used:
                selected.append(item)
                used.add(item)
            if len(selected) >= 10:
                break
        predictions[q] = selected[:10]
    output = []
    for pred in [data['native'], predictions]:
        values = []
        for (_, k, ndcg) in METRICS:
            hits = pred[:, :k] == data['targets'][:, None]
            values.append(float((hits / np.log2(np.arange(k) + 2)).sum(1).mean()) if ndcg else float(hits.any(1).mean()))
        output.append(values)
    for (side, values) in zip(['native', 'fused'], output):
        for (index, key) in [(3, 'recall@10'), (4, 'ndcg@10')]:
            np.testing.assert_allclose(values[index], result['test'][f'{side}_{key}'], rtol=0, atol=1e-12)
    return output
