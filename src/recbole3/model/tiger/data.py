from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from recbole3.dataset import ITEM_ID, USER_ID
from recbole3.config import instantiate_dataclass
from recbole3.model.base import BaseCollator, ModelConfig, ModelDatasets
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS
from recbole3.model.tiger.config import TIGERConfig


def _position_utilization_entropy(item_to_sid: dict[int, tuple[int, ...]], position: int) -> float:
    """Shannon entropy (nats) of the code-value distribution at ``position``. Low => collapsed."""
    counts: dict[int, int] = defaultdict(int)
    for sid in item_to_sid.values():
        counts[sid[position]] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def _resolve_recode_positions(item_to_sid: dict[int, tuple[int, ...]], spec: str, width: int) -> list[int]:
    """'auto' -> the single position with the largest *achievable* rebalance gain; else a parsed list.

    Informative = value-cardinality > 2 (skips constant markers and binary dedup/disambiguation
    positions). The most-collapsed position is the WRONG target: if it is globally collapsed its
    per-parent value pools are size 1, so the recode (which reuses each group's existing pool) is a
    no-op. We instead score each candidate by the entropy it would actually gain after recoding, and
    pick the max (0 gain -> nothing worth recoding)."""
    spec = str(spec or "").strip().lower()
    if spec and spec != "auto":
        return [int(p) for p in spec.split(",") if p.strip() != ""]
    best_pos, best_gain = None, 0.0
    for pos in range(width):
        if len({sid[pos] for sid in item_to_sid.values()}) <= 2:
            continue
        before = _position_utilization_entropy(item_to_sid, pos)
        after = _position_utilization_entropy(_recode_item_sids(item_to_sid, [pos]), pos)
        if after - before > best_gain:
            best_pos, best_gain = pos, after - before
    return [best_pos] if best_pos is not None else []


def _recode_item_sids(
    item_to_sid: dict[int, tuple[int, ...]],
    positions: list[int],
    embeddings: "np.ndarray | None" = None,
) -> dict[int, tuple[int, ...]]:
    """C4 recode: balance code-value usage at each target position, within each parent-prefix group,
    reusing only the value pool already present in that group. Guarantees item<->tuple uniqueness
    (items sharing a suffix get distinct values) and preserves the parent prefix + suffix (prefix tree
    stays valid for autoregressive decoding).

    Two reassignment rules:
      * ``embeddings is None`` — **entropy-only** (model-free): send each item to the least-used
        eligible code. Balances utilization but is semantics-blind (can scatter coherent clusters).
      * ``embeddings`` given — **signal-guided**: each code value gets a centroid = mean of its items'
        (frozen) embeddings; every item is sent to its *nearest under-capacity* centroid. Rebalances
        utilization while keeping semantically-close items on the same code (only boundary items move).
    """
    sids = dict(item_to_sid)
    for pos in positions:
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for item_id, sid in sids.items():
            groups[sid[:pos]].append(item_id)
        for _prefix, items in groups.items():
            pool = sorted({sids[i][pos] for i in items})
            if len(pool) <= 1:  # nothing to rebalance without inventing new codes
                continue
            cap = math.ceil(len(items) / len(pool))  # balanced-utilization capacity per code
            usage = {v: 0 for v in pool}
            used_by_suffix: dict[tuple[int, ...], set[int]] = defaultdict(set)
            centroid: dict[int, np.ndarray] | None = None
            if embeddings is not None:
                centroid = {}
                for v in pool:
                    vecs = np.stack([embeddings[i] for i in items if sids[i][pos] == v])
                    c = vecs.mean(0)
                    n = float(np.linalg.norm(c))
                    centroid[v] = c / n if n > 0 else c
            items.sort(key=lambda i: sids[i][pos + 1 :])  # suffix order: uniqueness determinism
            for item_id in items:
                sid = sids[item_id]
                suffix = sid[pos + 1 :]
                cands = [v for v in pool if v not in used_by_suffix[suffix]] or list(pool)
                if centroid is None:  # entropy-only: least-used eligible code
                    chosen = min(cands, key=lambda v: (usage[v], v))
                else:  # signal-guided: nearest under-capacity centroid (semantic-preserving)
                    e = embeddings[item_id]
                    chosen = min(cands, key=lambda v: (usage[v] >= cap, -float(np.dot(e, centroid[v]))))
                used_by_suffix[suffix].add(chosen)
                usage[chosen] += 1
                sids[item_id] = sid[:pos] + (chosen,) + suffix
    return sids


@dataclass(frozen=True, slots=True)
class TIGERSIDCodec:
    """Map RecBole item ids to TIGER semantic tokens and back."""

    item_to_sid: dict[int, tuple[int, ...]]
    item_to_tokens: dict[int, tuple[int, ...]]
    tokens_to_item: dict[tuple[int, ...], int]
    n_digit: int
    semantic_vocab_size: int
    fallback_item_ids: tuple[int, ...]

    @classmethod
    def from_file(
        cls,
        sid_file: str,
        *,
        num_items: int,
        recode_positions: str | None = None,
        recode_signal_file: str | None = None,
    ) -> "TIGERSIDCodec":
        if not str(sid_file or "").strip():
            raise ValueError("TIGERConfig.sid_file must point to an item_sids.json file.")
        path = Path(sid_file)
        if not path.exists():
            raise FileNotFoundError(f"TIGER sid_file does not exist: {path}")
        with open(path, "r", encoding="utf-8") as file:
            raw_sids = json.load(file)
        if not isinstance(raw_sids, dict):
            raise ValueError("TIGER item_sids.json must be a JSON object.")

        item_to_sid: dict[int, tuple[int, ...]] = {}
        expected_width: int | None = None
        max_sid = -1
        for raw_item_id, raw_sid in raw_sids.items():
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"TIGER SID key must be a remapped item_id string, got {raw_item_id!r}.") from exc
            if item_id < 0 or item_id >= num_items:
                raise ValueError(f"TIGER SID item_id {item_id} is outside dataset range [0, {num_items - 1}].")
            if not isinstance(raw_sid, list) or not raw_sid:
                raise ValueError(f"TIGER SID for item_id {item_id} must be a non-empty list.")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in raw_sid):
                raise ValueError(f"TIGER SID for item_id {item_id} must contain only non-negative integers.")
            sid = tuple(raw_sid)
            if any(value < 0 for value in sid):
                raise ValueError(f"TIGER SID for item_id {item_id} contains negative values: {sid}.")
            if expected_width is None:
                expected_width = len(sid)
            elif len(sid) != expected_width:
                raise ValueError(
                    "All TIGER SID lists must have the same length. "
                    f"Expected {expected_width}, got {len(sid)} for item_id {item_id}."
                )
            item_to_sid[item_id] = sid
            max_sid = max(max_sid, *sid)

        missing = sorted(set(range(num_items)) - set(item_to_sid))
        if missing:
            preview = ", ".join(str(item_id) for item_id in missing[:10])
            raise ValueError(f"TIGER SID file is missing {len(missing)} item ids. First missing ids: {preview}.")

        # C4 (codebook redistribution) — recode the item->SID assignment before tokenization.
        if recode_positions is not None and str(recode_positions).strip():
            positions = _resolve_recode_positions(item_to_sid, str(recode_positions), int(expected_width or 0))
            if positions:
                emb = None
                if recode_signal_file and str(recode_signal_file).strip():
                    emb = np.load(str(recode_signal_file)).astype(np.float32)
                    if emb.shape[0] < num_items:
                        raise ValueError(
                            f"recode_signal_file has {emb.shape[0]} rows < num_items {num_items}."
                        )
                before = {p: _position_utilization_entropy(item_to_sid, p) for p in positions}
                item_to_sid = _recode_item_sids(item_to_sid, positions, embeddings=emb)
                after = {p: _position_utilization_entropy(item_to_sid, p) for p in positions}
                print(
                    "[C4-recode] positions="
                    + ",".join(str(p) for p in positions)
                    + f" signal={os.path.basename(str(recode_signal_file)) if emb is not None else 'none'}"
                    + " utilization-entropy(nats) "
                    + "; ".join(f"pos{p}: {before[p]:.3f}->{after[p]:.3f}" for p in positions),
                    flush=True,
                )

        item_to_tokens = {
            item_id: tuple(value + 1 for value in sid)
            for item_id, sid in item_to_sid.items()
        }
        tokens_to_item: dict[tuple[int, ...], int] = {}
        for item_id, tokens in item_to_tokens.items():
            if tokens in tokens_to_item:
                raise ValueError(
                    "TIGER SID file contains duplicate SID tuples after offsetting. "
                    f"Items {tokens_to_item[tokens]} and {item_id} both map to {tokens}."
                )
            tokens_to_item[tokens] = item_id

        return cls(
            item_to_sid=item_to_sid,
            item_to_tokens=item_to_tokens,
            tokens_to_item=tokens_to_item,
            n_digit=int(expected_width or 0),
            semantic_vocab_size=max_sid + 1,
            fallback_item_ids=tuple(sorted(item_to_sid)),
        )

    def item_tokens(self, item_id: int) -> tuple[int, ...]:
        return self.item_to_tokens[int(item_id)]

    def token_tuple_to_item(self, tokens: Sequence[int]) -> int | None:
        return self.tokens_to_item.get(tuple(int(token) for token in tokens))


class TIGERModelDataset(BaseSequentialModelDataset):
    """Model-side dataset that adds TIGER prefix histories and SID mappings."""

    tiger_codec: TIGERSIDCodec

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets[pd.DataFrame, pd.DataFrame]:
        if not isinstance(model_config, TIGERConfig):
            model_config = instantiate_dataclass(TIGERConfig, model_config)
        _validate_num_beams(model_config)
        self.tiger_codec = TIGERSIDCodec.from_file(
            model_config.sid_file,
            num_items=int(self.get_num_items()),
            recode_positions=(model_config.recode_positions if model_config.recode_enabled else None),
            recode_signal_file=(model_config.recode_signal_file if model_config.recode_enabled else None),
        )
        return super()._build_model_datasets(model_config=model_config)


def _validate_num_beams(config: TIGERConfig) -> None:
    if int(config.num_beams) <= 0:
        raise ValueError("TIGERConfig.num_beams must be a positive integer.")
    topk = tuple(int(k) for k in config.eval_topk)
    if not topk:
        raise ValueError("TIGERConfig.eval_topk must contain at least one positive k.")
    if any(k <= 0 for k in topk):
        raise ValueError(f"TIGERConfig.eval_topk values must be positive, got {topk}.")
    max_topk = max(topk)
    if int(config.num_beams) < max_topk:
        raise ValueError(
            f"TIGERConfig.num_beams ({config.num_beams}) must be >= max(eval_topk) ({max_topk}). "
            "Keep model.eval_topk aligned with trainer.eval.metrics."
        )


class _TIGERBaseCollator(BaseCollator):
    config: TIGERConfig

    def __init__(self, config: TIGERConfig, prepared_data: TIGERModelDataset):
        super().__init__(config, prepared_data)
        if not hasattr(prepared_data, "tiger_codec"):
            raise RuntimeError("TIGER prepared_data is missing tiger_codec. Use TIGERModelDataset as model_data_cls.")
        self.codec = prepared_data.tiger_codec
        self.pad_token = 0
        if int(config.n_user_tokens) <= 0:
            raise ValueError("TIGERConfig.n_user_tokens must be a positive integer.")
        self.base_user_token = self.codec.semantic_vocab_size + 1
        self.eos_token = self.base_user_token + int(config.n_user_tokens)
        self.history_max_length = _normalize_history_max_length(config.history_max_length)
        self.user_signatures = _load_user_signatures(
            config.user_signature_file,
            min_items=int(config.user_signature_min_items),
            semantic_vocab_size=int(self.codec.semantic_vocab_size),
            disjoint=bool(config.user_signature_disjoint),
            base_user_token=int(self.base_user_token),
            n_user_tokens=int(config.n_user_tokens),
        )
        self.max_signature_tokens = max((len(tokens) for tokens in self.user_signatures.values()), default=0)
        # +2 = hashed-or-absent user slot budget replaced by up to max_signature_tokens, plus eos.
        # Keep at least the original "+2" (1 user token + eos) when signatures are unused.
        prefix_budget = max(1, self.max_signature_tokens) + 1
        self.max_token_seq_len = (
            None
            if self.history_max_length is None
            else self.history_max_length * self.codec.n_digit + prefix_budget
        )
        self.literal_tokenizer = None
        self.literal_token_text: dict[int, str] = {}
        literal_path = str(config.literal_text_file or "").strip()
        if literal_path:
            if self.user_signatures:
                raise ValueError("literal-text encoder currently supports hashed user prefixes only.")
            path = Path(literal_path)
            if not path.exists():
                raise FileNotFoundError(f"TIGER literal_text_file does not exist: {path}")
            with path.open(encoding="utf-8") as handle:
                self.literal_token_text = {
                    int(token): str(phrase or "").strip() for token, phrase in json.load(handle).items()
                }
            from transformers import AutoTokenizer

            self.literal_tokenizer = AutoTokenizer.from_pretrained(config.literal_text_model)

    def _user_token(self, user_id: int) -> int:
        return self.base_user_token + int(user_id) % int(self.config.n_user_tokens)

    def _user_prefix(self, user_id: int) -> list[int]:
        signature = self.user_signatures.get(int(user_id))
        if signature:
            return list(signature)
        return [self._user_token(user_id)]

    def _input_for_record(self, record: Mapping[str, Any]) -> tuple[list[int], list[int]]:
        history = tuple(int(item_id) for item_id in (record.get(HISTORY_ITEM_IDS) or ()))
        if self.history_max_length is None:
            pass
        elif self.history_max_length > 0:
            history = history[-self.history_max_length:]
        else:
            history = ()

        input_ids = self._user_prefix(int(record[USER_ID]))
        for item_id in history:
            input_ids.extend(self.codec.item_tokens(item_id))
        input_ids.append(self.eos_token)

        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask

    def _literal_input_for_record(self, record: Mapping[str, Any]) -> tuple[list[int], list[int]] | None:
        if self.literal_tokenizer is None:
            return None
        history = tuple(int(item_id) for item_id in (record.get(HISTORY_ITEM_IDS) or ()))
        if self.history_max_length is not None:
            history = history[-self.history_max_length:] if self.history_max_length > 0 else ()
        token_ids: list[int] = []
        separator = int(self.literal_tokenizer.eos_token_id)
        for item_id in history:
            phrases = [
                self.literal_token_text[token]
                for token in self.codec.item_tokens(item_id)
                if self.literal_token_text.get(token)
            ]
            if phrases:
                token_ids.extend(self.literal_tokenizer.encode(" ".join(phrases), add_special_tokens=False))
                token_ids.append(separator)
        if not token_ids:
            token_ids = [separator]
        max_tokens = int(self.config.literal_text_max_tokens)
        if max_tokens <= 0:
            raise ValueError("literal_text_max_tokens must be positive.")
        token_ids = token_ids[-max_tokens:]
        return token_ids, [1] * len(token_ids)

    def _add_literal_inputs(
        self, batch: dict[str, torch.Tensor], records: Sequence[Mapping[str, Any]]
    ) -> dict[str, torch.Tensor]:
        rows = [self._literal_input_for_record(record) for record in records]
        if not rows or rows[0] is None:
            return batch
        literal_rows = [row for row in rows if row is not None]
        max_len = max(len(row[0]) for row in literal_rows)
        pad = int(self.literal_tokenizer.pad_token_id)
        ids, masks = [], []
        for token_ids, mask in literal_rows:
            width = max_len - len(token_ids)
            ids.append(token_ids + [pad] * width)
            masks.append(mask + [0] * width)
        batch["literal_input_ids"] = torch.tensor(ids, dtype=torch.long)
        batch["literal_attention_mask"] = torch.tensor(masks, dtype=torch.long)
        return batch

    def _pad_inputs(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        *,
        max_token_seq_len: int,
    ) -> tuple[list[int], list[int]]:
        pad_width = max_token_seq_len - len(input_ids)
        if pad_width < 0:
            raise ValueError("TIGER input sequence exceeded max_token_seq_len.")
        input_ids.extend([self.pad_token] * pad_width)
        attention_mask.extend([0] * pad_width)
        return input_ids, attention_mask

    def _batch_max_token_seq_len(self, input_ids: Sequence[Sequence[int]]) -> int:
        if self.max_token_seq_len is not None:
            return self.max_token_seq_len
        # None follows SequentialModelConfig semantics: keep unbounded history.
        # Padding is therefore computed from the longest record in the current batch.
        return max((len(row) for row in input_ids), default=2)

    def _records(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if isinstance(feature_records, pd.DataFrame):
            return feature_records.to_dict("records")
        return list(feature_records)


class TIGERTrainCollator(_TIGERBaseCollator):
    """Collate sequential training records into T5 teacher-forcing batches."""

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(feature_records)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        user_ids: list[int] = []
        for record in rows:
            cur_input_ids, cur_attention_mask = self._input_for_record(record)
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            labels.append(list(self.codec.item_tokens(int(record[ITEM_ID]))) + [self.eos_token])
            user_ids.append(int(record[USER_ID]))
        max_token_seq_len = self._batch_max_token_seq_len(input_ids)
        padded = [
            self._pad_inputs(cur_input_ids, cur_attention_mask, max_token_seq_len=max_token_seq_len)
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask, strict=False)
        ]
        batch = {
            "input_ids": torch.tensor([row[0] for row in padded], dtype=torch.long),
            "attention_mask": torch.tensor([row[1] for row in padded], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "user_ids": torch.tensor(user_ids, dtype=torch.long),
        }
        if self.config.path_ranking_enabled:
            histories = [list(record.get(HISTORY_ITEM_IDS) or ()) for record in rows]
            max_history = max((len(history) for history in histories), default=0)
            batch["ranking_seen_item_ids"] = torch.tensor(
                [history + [-1] * (max_history - len(history)) for history in histories],
                dtype=torch.long,
            )
        return self._add_literal_inputs(batch, rows)


class TIGEREvalCollator(_TIGERBaseCollator):
    """Collate evaluation records into T5 generation inputs."""

    def __call__(self, feature_records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = self._records(feature_records)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        user_ids: list[int] = []
        for record in rows:
            cur_input_ids, cur_attention_mask = self._input_for_record(record)
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            user_ids.append(int(record[USER_ID]))
        max_token_seq_len = self._batch_max_token_seq_len(input_ids)
        padded = [
            self._pad_inputs(cur_input_ids, cur_attention_mask, max_token_seq_len=max_token_seq_len)
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask, strict=False)
        ]
        batch = {
            "input_ids": torch.tensor([row[0] for row in padded], dtype=torch.long),
            "attention_mask": torch.tensor([row[1] for row in padded], dtype=torch.long),
            "user_ids": torch.tensor(user_ids, dtype=torch.long),
        }
        return self._add_literal_inputs(batch, rows)


def _normalize_history_max_length(history_max_length: int | None) -> int | None:
    if history_max_length is None:
        return None
    return int(history_max_length)


def _load_user_signatures(
    signature_file: str,
    *,
    min_items: int,
    semantic_vocab_size: int,
    disjoint: bool = False,
    base_user_token: int | None = None,
    n_user_tokens: int | None = None,
) -> dict[int, tuple[int, ...]]:
    """Load eligible offline user signatures as TIGER token ids.

    Default (``disjoint=False``): signature codes reuse the item token range, ``code -> code+1``
    (validated ``0 <= code < semantic_vocab_size``). With ``disjoint=True`` the codes are mapped
    into the dedicated user-token band, ``code -> base_user_token + code`` (validated
    ``0 <= code < n_user_tokens``), so they never collide with item SID tokens.
    """
    if not str(signature_file or "").strip():
        return {}
    if min_items <= 0:
        raise ValueError("TIGERConfig.user_signature_min_items must be a positive integer.")
    if disjoint:
        if base_user_token is None or n_user_tokens is None:
            raise ValueError("Disjoint user signatures require base_user_token and n_user_tokens.")
        if int(n_user_tokens) <= 0:
            raise ValueError("TIGERConfig.n_user_tokens must be a positive integer for disjoint signatures.")
    path = Path(signature_file)
    if not path.exists():
        raise FileNotFoundError(f"TIGER user_signature_file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    users = payload.get("users") if isinstance(payload, dict) else None
    if not isinstance(users, dict):
        raise ValueError("TIGER user_signature_file must contain a top-level 'users' object.")

    signatures: dict[int, tuple[int, ...]] = {}
    for raw_user_id, row in users.items():
        if not isinstance(row, dict):
            raise ValueError(f"User signature for {raw_user_id!r} must be an object.")
        n_train = int(row.get("n_train_unique_items", 0))
        if n_train < min_items:
            continue
        codes = row.get("signature_codes")
        if not isinstance(codes, list) or not codes:
            raise ValueError(f"User {raw_user_id!r} is missing a non-empty signature_codes list.")
        if any(not isinstance(code, int) or isinstance(code, bool) or int(code) < 0 for code in codes):
            raise ValueError(f"User {raw_user_id!r} signature_codes must be non-negative integers.")
        if disjoint:
            if any(int(code) >= int(n_user_tokens) for code in codes):
                raise ValueError(
                    f"User {raw_user_id!r} disjoint signature code exceeds n_user_tokens={n_user_tokens}. "
                    "Raise model.n_user_tokens to the builder's required_n_user_tokens."
                )
            # Land in the dedicated user-token band; no collision with item SID tokens.
            signatures[int(raw_user_id)] = tuple(int(base_user_token) + int(code) for code in codes)
        else:
            if any(int(code) >= semantic_vocab_size for code in codes):
                raise ValueError(
                    f"User {raw_user_id!r} signature code exceeds semantic_vocab_size={semantic_vocab_size}."
                )
            # Same offset as item SID tokens (pad=0, code c -> token c+1).
            signatures[int(raw_user_id)] = tuple(int(code) + 1 for code in codes)
    return signatures


__all__ = [
    "TIGEREvalCollator",
    "TIGERModelDataset",
    "TIGERSIDCodec",
    "TIGERTrainCollator",
]
