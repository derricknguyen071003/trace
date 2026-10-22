from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.sequential import HISTORY_ITEM_IDS
from recbole3.model.tiger.config import TIGERConfig
from recbole3.model.tiger.data import (
    TIGEREvalCollator,
    TIGERModelDataset,
    TIGERSIDCodec,
    TIGERTrainCollator,
    _load_user_signatures,
)


# ---------------------------------------------------------------------------
# In-loop running early-exit (deployable C3): halt generate() at the depth the
# running rule settles, banking real wall-clock. Item names for a committed
# depth-d prefix come from a static prefix->item table built offline from the
# frozen SID assignment (a free lookup, NOT runtime full-depth decoding).
# ---------------------------------------------------------------------------


def _necessity_jaccard(cand: list[int], ref: list[int]) -> float:
    a, b = set(cand), set(ref)
    union = len(a | b)
    return 1.0 - ((len(a & b) / union) if union else 1.0)


def _necessity_ndcg(cand: list[int], ref: list[int]) -> float:
    if not ref:
        return 0.0
    gain = {item: 1.0 / math.log2(r + 2) for r, item in enumerate(ref)}
    idcg = sum(gain[ref[j]] / math.log2(j + 2) for j in range(len(ref)))
    if idcg <= 0:
        return 0.0
    dcg = sum(gain.get(cand[j], 0.0) / math.log2(j + 2) for j in range(len(cand)))
    return max(0.0, 1.0 - dcg / idcg)


def _build_prefix_resolver(codec: "TIGERSIDCodec") -> dict[tuple[int, ...], int]:
    """Map every code prefix (all depths) to a representative item from the frozen SID table.

    Representative = the item whose full SID is lexicographically smallest among items sharing
    that prefix. Deterministic and query-independent; at the running stop depth the surviving
    prefixes are ~unique so this choice matches what full decoding would name.
    """
    resolver: dict[tuple[int, ...], int] = {}
    for item_id in sorted(codec.item_to_tokens):
        toks = codec.item_to_tokens[item_id]
        for depth in range(1, len(toks) + 1):
            pref = tuple(int(t) for t in toks[:depth])
            cur = resolver.get(pref)
            if cur is None or toks < codec.item_to_tokens[cur]:
                resolver[pref] = item_id
    return resolver


def _select_topk_from_prefixes(
    prefixes: list[tuple[int, ...]],
    excl: set[int],
    resolver: dict[tuple[int, ...], int],
    fallback: tuple[int, ...],
    k: int,
) -> list[int]:
    """Top-K commit set at a fixed depth: dedup beams by their depth-d prefix bucket (beam order
    == score order), name each bucket via the static resolver, pad with fallback to exactly K."""
    seen_pref: set[tuple[int, ...]] = set()
    out: list[int] = []
    out_set: set[int] = set()
    for pref in prefixes:
        if pref in seen_pref:
            continue
        seen_pref.add(pref)
        item = resolver.get(pref)
        if item is None or item in excl or item in out_set:
            continue
        out.append(item)
        out_set.add(item)
        if len(out) == k:
            break
    if len(out) < k:
        for item in fallback:
            if item in out_set or item in excl:
                continue
            out.append(item)
            out_set.add(item)
            if len(out) == k:
                break
    return out


def _running_inloop_scan(
    beam_tuples: list[tuple[int, ...]],
    excl: set[int],
    resolver: dict[tuple[int, ...], int],
    fallback: tuple[int, ...],
    *,
    k: int,
    n_digit: int,
    min_depth: int,
    stable_m: int,
    tau: float,
    metric,
) -> tuple[list[int], int]:
    """Pure reference for the in-loop running rule (also the test oracle): return (topK, stop_depth).

    Identical decision to the incremental StoppingCriteria — walks depth upward, stops the first
    time topK@d has been stable vs topK@(d-1) for ``stable_m`` consecutive depths."""
    md = max(1, min(int(min_depth), int(n_digit)))
    prev: list[int] | None = None
    stable = 0
    for d in range(md, int(n_digit) + 1):
        prefixes = [tuple(t[:d]) for t in beam_tuples]
        cur = _select_topk_from_prefixes(prefixes, excl, resolver, fallback, k)
        if prev is not None and metric(cur, prev) <= tau:
            stable += 1
            if stable >= max(1, int(stable_m)):
                return cur, d
        else:
            stable = 0
        prev = cur
    return (prev if prev is not None else []), int(n_digit)


_RUNNING_STOP_CLS = None


def _running_stop_cls():
    """Lazily build the StoppingCriteria subclass (keeps transformers import out of module load)."""
    global _RUNNING_STOP_CLS
    if _RUNNING_STOP_CLS is not None:
        return _RUNNING_STOP_CLS
    from transformers import StoppingCriteria

    class _RunningTopKStop(StoppingCriteria):
        """Applies the running rule inside generate(); returns all-True to halt once every example
        in the batch has settled (or full depth reached). Commits each example at its own depth."""

        def __init__(self, *, resolver, excluded, fallback, n_digit, k, num_beams, batch_size,
                     min_depth, stable_m, tau, metric):
            self.resolver = resolver
            self.excluded = excluded
            self.fallback = fallback
            self.n_digit = int(n_digit)
            self.k = int(k)
            self.num_beams = int(num_beams)
            self.batch_size = int(batch_size)
            self.min_depth = max(1, min(int(min_depth), self.n_digit))
            self.stable_m = max(1, int(stable_m))
            self.tau = float(tau)
            self.metric = metric
            self.prev = [None] * self.batch_size
            self.stable = [0] * self.batch_size
            self.done = [False] * self.batch_size
            self.committed = [None] * self.batch_size
            self.committed_depth = [self.n_digit] * self.batch_size

        def __call__(self, input_ids, scores, **kwargs):
            total = int(input_ids.shape[0])
            cur_len = int(input_ids.shape[1])
            depth = cur_len - 1  # code tokens generated so far (col 0 is decoder_start)
            if depth >= self.min_depth:
                d = min(depth, self.n_digit)
                # Beam-search internals can carry more live rows than the requested
                # ``num_beams`` (for example during grouped candidate expansion).
                # ``generate`` passes those rows to stopping criteria before reducing
                # them to returned beams. Infer the live beam count from input_ids.
                if total % self.batch_size:
                    raise RuntimeError(
                        "C3 stopping criteria received rows not divisible by batch size: "
                        f"rows={total}, batch_size={self.batch_size}."
                    )
                live_beams = total // self.batch_size
                ids = input_ids.view(self.batch_size, live_beams, cur_len)
                for b in range(self.batch_size):
                    if self.done[b]:
                        continue
                    excl = self.excluded[b]
                    prefixes = [
                        tuple(int(t) for t in ids[b, beam, 1 : 1 + d].tolist())
                        for beam in range(live_beams)
                    ]
                    cur = _select_topk_from_prefixes(prefixes, excl, self.resolver, self.fallback, self.k)
                    if self.prev[b] is not None and self.metric(cur, self.prev[b]) <= self.tau:
                        self.stable[b] += 1
                        if self.stable[b] >= self.stable_m:
                            self.done[b] = True
                            self.committed[b] = cur
                            self.committed_depth[b] = d
                    elif self.prev[b] is not None:
                        self.stable[b] = 0
                    self.prev[b] = cur
                if d >= self.n_digit:
                    for b in range(self.batch_size):
                        if not self.done[b]:
                            self.done[b] = True
                            self.committed[b] = self.prev[b]
                            self.committed_depth[b] = self.n_digit
            halt = all(self.done)
            return torch.full((total,), bool(halt), dtype=torch.bool, device=input_ids.device)

    _RUNNING_STOP_CLS = _RunningTopKStop
    return _RUNNING_STOP_CLS


class _CFSoftPrompt(nn.Module):
    """Project per-user continuous CF profiles into T5 encoder prefix embeddings (Option 3).

    Holds a dense ``[num_users, d_cf]`` profile buffer (missing / cold users are zeroed,
    which after the builder's standardization equals the mean profile). A small learned MLP
    maps each profile to ``n_tokens`` continuous prefix embeddings of width ``d_model``.
    """

    def __init__(
        self,
        profiles: torch.Tensor,
        *,
        d_model: int,
        n_tokens: int,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_tokens <= 0:
            raise ValueError("cf_soft_prompt_tokens must be a positive integer.")
        self.register_buffer("profiles", profiles, persistent=False)
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)
        self.feature_dropout = nn.Dropout(float(dropout))
        self.mlp = nn.Sequential(
            nn.Linear(int(profiles.shape[1]), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.n_tokens * self.d_model),
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        idx = user_ids.to(self.profiles.device).long().clamp_(0, self.profiles.shape[0] - 1)
        profile = self.feature_dropout(self.profiles[idx])  # [B, d_cf]
        prefix = self.mlp(profile)  # [B, n_tokens * d_model]
        return prefix.view(prefix.shape[0], self.n_tokens, self.d_model)


class _CFAlignHead(nn.Module):
    """Auxiliary LETTER-style CF alignment for the USER side (no added token).

    Holds a dense ``[num_users, d_cf]`` CF-profile buffer plus a per-user eligibility mask.
    A small MLP projects the encoder's mean-pooled history representation into CF space; an
    InfoNCE loss pulls each (eligible) user's projected representation toward that user's own
    CF profile and away from the other eligible users' profiles in the batch. CF enters only
    through this loss — no vocabulary, no prefix, no token-budget cost.
    """

    def __init__(
        self,
        profiles: torch.Tensor,
        eligible: torch.Tensor,
        *,
        d_model: int,
        hidden: int,
        temp: float,
    ) -> None:
        super().__init__()
        self.register_buffer("profiles", profiles, persistent=False)
        self.register_buffer("eligible", eligible, persistent=False)
        self.temp = max(float(temp), 1e-4)
        d_cf = int(profiles.shape[1])
        self.proj = nn.Sequential(
            nn.Linear(int(d_model), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), d_cf),
        )

    def forward(self, pooled: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        idx = user_ids.to(self.profiles.device).long().clamp_(0, self.profiles.shape[0] - 1)
        mask = self.eligible[idx]  # [B] bool
        # Restrict the contrastive set to eligible users only: their profiles are non-zero
        # (standardized), so cosine normalization is well-defined; ineligible users are zeroed.
        if int(mask.sum()) < 2:
            return pooled.new_zeros(())
        z = self.proj(pooled[mask].float())  # [M, d_cf]
        z = nn.functional.normalize(z, dim=-1)
        e = nn.functional.normalize(self.profiles[idx][mask], dim=-1)  # [M, d_cf]
        logits = (z @ e.t()) / self.temp  # [M, M]
        labels = torch.arange(z.shape[0], device=z.device)
        return nn.functional.cross_entropy(logits, labels)


class TIGERModel(BaseRetrievalModel):
    """TIGER generative retrieval model backed by a small T5 decoder target space."""

    config: TIGERConfig

    def __init__(self, config: TIGERConfig):
        super().__init__(config)
        self._t5: torch.nn.Module | None = None
        self._soft_prompt: _CFSoftPrompt | None = None
        self._soft_prompt_tokens = 0
        self._align_head: _CFAlignHead | None = None
        self._literal_text_embedding: torch.nn.Module | None = None
        self._literal_text_projection: torch.nn.Module | None = None
        self._codec: TIGERSIDCodec | None = None
        self._pad_token = 0
        self._base_user_token: int | None = None
        self._eos_token: int | None = None
        self._vocab_size: int | None = None
        self._max_observed_history_items: int | None = None
        self._max_signature_tokens = 0
        # Frozen lexical C1 keeps only phrase-backed SID rows fixed. These buffers
        # move with the model but are inference-irrelevant and need not enter a checkpoint.
        self.register_buffer("_frozen_text_row_mask", None, persistent=False)
        self.register_buffer("_frozen_text_row_values", None, persistent=False)
        # Lexical-anchor buffers retain the projected phrase initialization. They are used
        # only during training and do not change the tokenizer or decoder targets.
        self.register_buffer("_text_anchor_row_mask", None, persistent=False)
        self.register_buffer("_text_anchor_row_values", None, persistent=False)
        self.register_buffer("_text_anchor_row_weights", None, persistent=False)
        self._frozen_text_grad_hook = None
        self._parent_prefix_children: tuple[dict[tuple[int, ...], tuple[int, ...]], ...] = ()
        self.register_parameter("path_ranking_log_scale", None)
        if config.parent_prefix_weight_mode not in {"fixed", "learned"}:
            raise ValueError("parent_prefix_weight_mode must be fixed or learned")
        self.register_parameter("phrase_log_scale", None)
        self._phrase_projection = None
        self.register_buffer("_phrase_targets", None, persistent=False)
        self.register_buffer("_phrase_mask", None, persistent=False)
        if config.phrase_loss_enabled:
            if config.phrase_loss_weight_mode not in {"fixed", "learned"} or config.phrase_loss_weight < 0:
                raise ValueError("Invalid phrase loss weight configuration")
            if not config.text_embed_file or config.text_embed_mode == "off":
                raise ValueError("Phrase loss requires enabled text embeddings")
            if config.phrase_loss_weight_mode == "learned":
                self.phrase_log_scale = nn.Parameter(torch.zeros(()))
        self.register_parameter("parent_prefix_log_scale", None)
        if config.parent_prefix_weight_mode == "learned":
            self.parent_prefix_log_scale = nn.Parameter(torch.zeros(()))
        if config.path_ranking_enabled:
            if config.path_ranking_weight_mode not in {"learned", "fixed"}:
                raise ValueError("path_ranking_weight_mode must be learned or fixed")
            if config.path_ranking_negatives < 1 or config.path_ranking_temperature <= 0 or config.path_ranking_weight < 0:
                raise ValueError("Invalid path ranking settings")
            if config.path_ranking_weight_mode == "learned":
                self.path_ranking_log_scale = nn.Parameter(torch.zeros(()))
        self._ranking_buckets: dict[tuple[int, ...], list[tuple[int, ...]]] = {}

    def build_train_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return TIGERTrainCollator(self.config, prepared_data)

    def build_eval_collator(self, prepared_data) -> BaseCollator:
        self._ensure_initialized(prepared_data)
        return TIGEREvalCollator(self.config, prepared_data)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        t5 = self._t5_module()
        encoder_kwargs = self._encoder_kwargs(
            batch["input_ids"], batch["attention_mask"], batch.get("user_ids"),
            batch.get("literal_input_ids"), batch.get("literal_attention_mask"),
        )
        phrase_enabled = self.training and self.config.phrase_loss_enabled
        outputs = t5(labels=batch.get("labels"), output_hidden_states=phrase_enabled, **encoder_kwargs)
        loss = outputs.loss
        phrase_loss = None
        phrase_weight = None
        if phrase_enabled and loss is not None:
            labels = batch["labels"][:, :self._codec.n_digit]
            valid = labels.ge(0) & labels.lt(self._phrase_mask.numel())
            safe_labels = labels.clamp(0, self._phrase_mask.numel() - 1)
            valid = valid & self._phrase_mask[safe_labels]
            if valid.any():
                states = outputs.decoder_hidden_states[-1][:, :labels.shape[1]]
                predicted = self._phrase_projection(states[valid])
                targets = self._phrase_targets[safe_labels[valid]]
                phrase_loss = (1 - torch.nn.functional.cosine_similarity(predicted, targets, dim=-1)).mean()
                if self.phrase_log_scale is not None:
                    scale = self.phrase_log_scale.clamp(-6.0, 6.0)
                    phrase_weight = torch.exp(-scale)
                    loss = loss + phrase_weight * phrase_loss + scale
                else:
                    phrase_weight = loss.new_tensor(self.config.phrase_loss_weight)
                    loss = loss + phrase_weight * phrase_loss
                self._phrase_steps = getattr(self, "_phrase_steps", 0) + 1
                if self._phrase_steps % 100 == 0:
                    print(f"[phrase-loss] loss={phrase_loss.detach().item():.6f} weight={phrase_weight.detach().item():.6f}", flush=True)
        if loss is not None:
            anchor_loss = self._text_anchor_loss()
            if anchor_loss is not None:
                loss = loss + float(self.config.text_anchor_weight) * anchor_loss
        if self._align_head is not None and loss is not None:
            user_ids = batch.get("user_ids")
            if user_ids is None:
                raise ValueError("CF alignment is enabled but the batch is missing 'user_ids'.")
            enc = outputs.encoder_last_hidden_state  # [B, L, d_model]
            enc_mask = encoder_kwargs["attention_mask"].to(enc.dtype).unsqueeze(-1)  # [B, L, 1]
            pooled = (enc * enc_mask).sum(1) / enc_mask.sum(1).clamp_min(1.0)  # [B, d_model]
            align_loss = self._align_head(pooled, user_ids)
            loss = loss + float(self.config.cf_align_weight) * align_loss
        survival_loss = None
        parent_prefix_loss = None
        parent_prefix_weight = float(self.config.parent_prefix_loss_weight)
        if parent_prefix_weight < 0:
            raise ValueError("parent_prefix_loss_weight must be non-negative.")
        parent_effective_weight = None
        learned_parent = self.parent_prefix_log_scale is not None
        if loss is not None and (parent_prefix_weight > 0 or (learned_parent and self.training)):
            labels = batch.get("labels")
            if labels is None:
                raise ValueError("Parent-prefix supervision requires training labels.")
            parent_prefix_loss = self._parent_prefix_loss(outputs.logits, labels)
            if learned_parent and self.training:
                scale = self.parent_prefix_log_scale.clamp(-6.0, 6.0)
                parent_effective_weight = torch.exp(-scale)
                # No evidence to learn a task scale from an all-singleton batch.
                if parent_prefix_loss.detach().item() > 0:
                    loss = loss + parent_effective_weight * parent_prefix_loss + scale
                self._parent_weight_steps = getattr(self, "_parent_weight_steps", 0) + 1
                if self._parent_weight_steps % 100 == 0:
                    print(f"[parent-prefix] loss={parent_prefix_loss.detach().item():.6f} weight={parent_effective_weight.detach().item():.6f}", flush=True)
            elif not learned_parent:
                parent_effective_weight = loss.new_tensor(parent_prefix_weight)
                loss = loss + parent_prefix_weight * parent_prefix_loss
        if loss is not None and float(self.config.prefix_survival_weight) > 0:
            labels = batch.get("labels")
            if labels is None:
                raise ValueError("Prefix-survival supervision requires training labels.")
            survival_loss = self._prefix_survival_loss(
                encoder_kwargs=encoder_kwargs,
                labels=labels,
                positive_logits=outputs.logits,
            )
            loss = loss + float(self.config.prefix_survival_weight) * survival_loss
        ranking_loss = None
        ranking_weight = None
        if self.training and loss is not None and self.config.path_ranking_enabled:
            ranking_loss = self._path_ranking_loss(batch, outputs, encoder_kwargs["attention_mask"])
            if self.path_ranking_log_scale is not None:
                # The additive scale penalty prevents the trivial weight -> 0 solution.
                scale = self.path_ranking_log_scale.clamp(-6.0, 6.0)
                ranking_weight = torch.exp(-scale)
                loss = loss + ranking_weight * ranking_loss + scale
            else:
                ranking_weight = loss.new_tensor(self.config.path_ranking_weight)
                loss = loss + ranking_weight * ranking_loss
            self._ranking_steps = getattr(self, "_ranking_steps", 0) + 1
            if self._ranking_steps % 100 == 0:
                print(f"[path-ranking] loss={ranking_loss.detach().item():.6f} weight={ranking_weight.detach().item():.6f}", flush=True)
        return {
            "loss": loss,
            "logits": outputs.logits,
            "outputs": outputs,
            "parent_prefix_loss": parent_prefix_loss,
            "parent_prefix_effective_weight": parent_effective_weight,
            "prefix_survival_loss": survival_loss,
            "path_ranking_loss": ranking_loss,
            "path_ranking_weight": ranking_weight,
            "phrase_loss": phrase_loss,
            "phrase_loss_weight": phrase_weight,
        }

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, Any]) -> torch.Tensor:
        del batch
        loss = outputs["loss"]
        if loss is None:
            raise ValueError("TIGER forward outputs did not include a loss. Training batches must include labels.")
        return loss

    def predict(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_item_ids is not None:
            raise NotImplementedError("TIGER phase 1 supports only full evaluation, not sampled evaluation.")

        t5 = self._t5_module()
        device = next(t5.parameters()).device
        input_ids = model_inputs["input_ids"].to(device)
        batch_size = int(input_ids.shape[0])
        if k <= 0:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        if int(self.config.num_beams) < int(k):
            raise ValueError(
                f"TIGERConfig.num_beams ({self.config.num_beams}) must be >= requested top-k ({k}). "
                "Increase model.num_beams or lower eval top-k."
            )

        codec = self._sid_codec()
        attention_mask = model_inputs["attention_mask"].to(device)
        beam_width = int(self.config.num_beams)

        user_ids = model_inputs.get("user_ids")
        if user_ids is not None:
            user_ids = user_ids.to(device)
        literal_input_ids = model_inputs.get("literal_input_ids")
        literal_attention_mask = model_inputs.get("literal_attention_mask")
        if literal_input_ids is not None:
            literal_input_ids = literal_input_ids.to(device)
        if literal_attention_mask is not None:
            literal_attention_mask = literal_attention_mask.to(device)
        encoder_kwargs = self._encoder_kwargs(
            input_ids, attention_mask, user_ids, literal_input_ids, literal_attention_mask
        )

        # Deployable in-loop running early-exit: halt generation at the settled depth (banks
        # wall-clock). Opt-in and gated by the running mode; leaves every other path unchanged.
        if (
            bool(self.config.adaptive_depth)
            and str(getattr(self.config, "adaptive_stop", "oracle")).lower() == "running"
            and bool(getattr(self.config, "adaptive_inloop", False))
        ):
            excluded_inloop = self._excluded_item_sets(exclude_item_ids, exclude_mask, batch_size=batch_size)
            return self._predict_running_inloop(
                encoder_kwargs=encoder_kwargs,
                codec=codec,
                excluded=excluded_inloop,
                k=k,
                beam_width=beam_width,
                batch_size=batch_size,
                device=input_ids.device,
            )

        generated = t5.generate(
            max_new_tokens=codec.n_digit + 1,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
            **encoder_kwargs,
        )
        sequences = generated.sequences.reshape(batch_size, beam_width, -1)
        if int(sequences.shape[-1]) < codec.n_digit + 1:
            raise ValueError(
                "TIGER generation returned sequences shorter than one SID tuple. "
                f"Got width {int(sequences.shape[-1])}, expected at least {codec.n_digit + 1}."
            )
        token_tuples = sequences[:, :, 1 : 1 + codec.n_digit]
        token_tuples_cpu = token_tuples.detach().cpu()
        excluded = self._excluded_item_sets(exclude_item_ids, exclude_mask, batch_size=batch_size)

        if bool(self.config.adaptive_depth):
            return self._predict_adaptive(
                token_tuples_cpu, excluded, codec, k=k, beam_width=beam_width, device=input_ids.device
            )

        predictions: list[list[int]] = []
        for row_index in range(batch_size):
            selected: list[int] = []
            selected_set: set[int] = set()
            for beam_index in range(beam_width):
                item_id = codec.token_tuple_to_item(token_tuples_cpu[row_index, beam_index].tolist())
                if item_id is None or item_id in selected_set or item_id in excluded[row_index]:
                    continue
                selected.append(item_id)
                selected_set.add(item_id)
                if len(selected) == k:
                    break
            if len(selected) < k:
                for item_id in codec.fallback_item_ids:
                    if item_id in selected_set or item_id in excluded[row_index]:
                        continue
                    selected.append(item_id)
                    selected_set.add(item_id)
                    if len(selected) == k:
                        break
            if len(selected) < k:
                raise ValueError(f"TIGER could not produce {k} predictions from {len(codec.fallback_item_ids)} items.")
            predictions.append(selected)

        return torch.tensor(predictions, dtype=torch.long, device=input_ids.device)

    def _prefix_resolver(self, codec: "TIGERSIDCodec") -> dict[tuple[int, ...], int]:
        """Static prefix->item table for the frozen SID assignment (built once, cached)."""
        cache = getattr(self, "_prefix_resolver_cache", None)
        if cache is None or cache[0] is not codec:
            cache = (codec, _build_prefix_resolver(codec))
            self._prefix_resolver_cache = cache
        return cache[1]

    def _predict_running_inloop(
        self,
        *,
        encoder_kwargs: Mapping[str, Any],
        codec: "TIGERSIDCodec",
        excluded: list[set[int]],
        k: int,
        beam_width: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Deployable C3: run generate() with the running-rule StoppingCriteria so it halts at the
        settled depth (real wall-clock saving). Item names come from the static prefix resolver, so
        no full-depth decode is needed to name a committed prefix."""
        from transformers import StoppingCriteriaList

        t5 = self._t5_module()
        resolver = self._prefix_resolver(codec)
        metric = _necessity_ndcg if str(getattr(self.config, "adaptive_metric", "jaccard")).lower() == "ndcg" else _necessity_jaccard
        stop = _running_stop_cls()(
            resolver=resolver,
            excluded=excluded,
            fallback=codec.fallback_item_ids,
            n_digit=int(codec.n_digit),
            k=int(k),
            num_beams=int(beam_width),
            batch_size=int(batch_size),
            min_depth=int(self.config.adaptive_min_depth),
            stable_m=int(getattr(self.config, "adaptive_stable_m", 1)),
            tau=float(self.config.adaptive_tau),
            metric=metric,
        )
        t5.generate(
            max_new_tokens=codec.n_digit + 1,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([stop]),
            **encoder_kwargs,
        )
        predictions: list[list[int]] = []
        for b in range(batch_size):
            selected = list(stop.committed[b] or [])
            if len(selected) < k:  # safety net: should not trigger (criteria pads to k)
                sel_set = set(selected)
                for item_id in codec.fallback_item_ids:
                    if item_id in sel_set or item_id in excluded[b]:
                        continue
                    selected.append(item_id)
                    sel_set.add(item_id)
                    if len(selected) == k:
                        break
            predictions.append(selected[:k])

        # Report the depth the decoder actually halted at (real wall-clock saving vs fixed L).
        L = int(codec.n_digit)
        depth_sum = sum(int(d) for d in stop.committed_depth)
        self._adaptive_depth_sum = getattr(self, "_adaptive_depth_sum", 0) + depth_sum
        self._adaptive_query_count = getattr(self, "_adaptive_query_count", 0) + batch_size
        mean_depth = self._adaptive_depth_sum / max(self._adaptive_query_count, 1)
        print(
            f"[adaptive-depth:running-inloop] batch_mean_depth={depth_sum / max(batch_size, 1):.3f} "
            f"cum_mean_depth={mean_depth:.3f}/{L} "
            f"cum_step_reduction={100.0 * (1.0 - mean_depth / L):.1f}% "
            f"(tau={float(self.config.adaptive_tau)}, m={int(getattr(self.config, 'adaptive_stable_m', 1))})",
            flush=True,
        )
        return torch.tensor(predictions, dtype=torch.long, device=device)

    def _predict_adaptive(
        self,
        token_tuples_cpu: torch.Tensor,
        excluded: list[set[int]],
        codec: "TIGERSIDCodec",
        *,
        k: int,
        beam_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Adaptive-depth (early-exit) decoding — efficiency C3, frozen model.

        Beams are already score-sorted by generate(). Truncating a beam tuple to depth d collapses
        beams sharing that coarse prefix into one bucket (dedup's collision counter is the last
        position, so shallow prefixes re-merge collided items); the top beam in a bucket is the
        tie-break representative. Two stop rules (``adaptive_stop``):

        * ``oracle`` — pick the shallowest d whose top-K matches the *full-depth* top-K within
          ``adaptive_tau`` (necessity = 1 - Jaccard(topK@d, topK@L)). Retrospective: it needs the
          full decode, so it is the (unrealizable) upper bound on step savings, not a wall-clock win.
        * ``running`` — deployable, peek-free: stop at the shallowest d where topK@d has been stable
          vs the previous depth (1 - Jaccard(topK@d, topK@(d-1)) <= tau) for ``adaptive_stable_m``
          consecutive depths. Never references L, so a real decoder can halt at d. Lands a little
          below the oracle (it pays one confirming step and can miss rare late churn).

        This is a retraining-free wrapper. The in-loop ``StoppingCriteria`` sibling
        (:class:`_RunningTopKStop`) applies the ``running`` rule inside generate() to bank wall-clock.
        """
        L = int(codec.n_digit)
        tau = float(self.config.adaptive_tau)
        min_depth = max(1, min(int(self.config.adaptive_min_depth), L))
        mode = str(getattr(self.config, "adaptive_stop", "oracle")).lower()
        stable_m = max(1, int(getattr(self.config, "adaptive_stable_m", 1)))
        metric = str(getattr(self.config, "adaptive_metric", "jaccard")).lower()
        batch_size = int(token_tuples_cpu.shape[0])

        def _necessity_jaccard(cand: list[int], ref: list[int]) -> float:
            a, b = set(cand), set(ref)
            union = len(a | b)
            return 1.0 - ((len(a & b) / union) if union else 1.0)

        def _necessity_ndcg(cand: list[int], ref: list[int]) -> float:
            """Order-AWARE: 1 - DCG(cand ordering scored by ref ranks) / DCG(ref). Treats the
            reference ranking (topK@L for oracle, topK@(d-1) for running) as ground truth: an item's
            gain is its rank-discount in ref, 0 if absent. Equals 0 iff cand reproduces ref's order
            for the items it contains; a pure reorder (which Jaccard ignores) makes it > 0."""
            if not ref:
                return 0.0
            gain = {item: 1.0 / math.log2(r + 2) for r, item in enumerate(ref)}
            idcg = sum(gain[ref[j]] / math.log2(j + 2) for j in range(len(ref)))
            if idcg <= 0:
                return 0.0
            dcg = sum(gain.get(cand[j], 0.0) / math.log2(j + 2) for j in range(len(cand)))
            return max(0.0, 1.0 - dcg / idcg)

        nec = _necessity_ndcg if metric == "ndcg" else _necessity_jaccard

        def select_at_depth(beams: list[list[int]], excl: set[int], depth: int) -> list[int]:
            seen_prefix: set[tuple[int, ...]] = set()
            out: list[int] = []
            for tup in beams:
                item = codec.token_tuple_to_item(tup)
                if item is None or item in excl:
                    continue
                prefix = tuple(tup[:depth])
                if prefix in seen_prefix:  # same depth-d bucket already has its top-beam representative
                    continue
                seen_prefix.add(prefix)
                out.append(item)
                if len(out) == k:
                    break
            if len(out) < k:  # fill exactly like the fixed-depth path so metrics stay comparable
                out_set = set(out)
                for item in codec.fallback_item_ids:
                    if item in out_set or item in excl:
                        continue
                    out.append(item)
                    out_set.add(item)
                    if len(out) == k:
                        break
            if len(out) < k:
                raise ValueError(f"TIGER adaptive decode could not produce {k} predictions.")
            return out

        def stop_oracle(beams: list[list[int]], excl: set[int]) -> tuple[list[int], int]:
            """Retrospective upper bound: compares each shallow depth to the FULL-depth top-K."""
            full = select_at_depth(beams, excl, L)
            for d in range(min_depth, L):
                cand = select_at_depth(beams, excl, d)
                if nec(cand, full) <= tau:
                    return cand, d
            return full, L

        def stop_running(beams: list[list[int]], excl: set[int]) -> tuple[list[int], int]:
            """Deployable, peek-free: stop once topK@d has been stable vs the previous depth for
            ``stable_m`` consecutive depths. Never references topK@L, so the decoder can halt at d."""
            prev_list = select_at_depth(beams, excl, min_depth)
            stable = 0
            for d in range(min_depth + 1, L + 1):
                cur = select_at_depth(beams, excl, d)
                if nec(cur, prev_list) <= tau:
                    stable += 1
                    if stable >= stable_m:
                        return cur, d  # decoded through d to confirm; d+1..L skipped
                else:
                    stable = 0
                prev_list = cur
            return prev_list, L  # never settled -> paid full depth

        step_fn = stop_running if mode == "running" else stop_oracle

        predictions: list[list[int]] = []
        depth_sum = 0
        for row in range(batch_size):
            beams = [token_tuples_cpu[row, b].tolist() for b in range(beam_width)]
            chosen, chosen_d = step_fn(beams, excluded[row])
            depth_sum += chosen_d
            predictions.append(chosen)

        # Running mean stop-depth => decode-step reduction vs fixed L. In 'oracle' mode this is the
        # (unrealizable) upper bound; in 'running' mode it is the depth the decoder can actually halt at.
        self._adaptive_depth_sum = getattr(self, "_adaptive_depth_sum", 0) + depth_sum
        self._adaptive_query_count = getattr(self, "_adaptive_query_count", 0) + batch_size
        mean_depth = self._adaptive_depth_sum / max(self._adaptive_query_count, 1)
        print(
            f"[adaptive-depth:{mode}] batch_mean_depth={depth_sum / max(batch_size, 1):.3f} "
            f"cum_mean_depth={mean_depth:.3f}/{L} "
            f"cum_step_reduction={100.0 * (1.0 - mean_depth / L):.1f}% (tau={tau}, m={stable_m}, metric={metric})",
            flush=True,
        )
        return torch.tensor(predictions, dtype=torch.long, device=device)

    def _ensure_initialized(self, prepared_data: TIGERModelDataset) -> None:
        if not hasattr(prepared_data, "tiger_codec"):
            raise RuntimeError("TIGERModel requires TIGERModelDataset prepared data.")
        codec = prepared_data.tiger_codec
        if self._codec is not None:
            if self._codec.n_digit != codec.n_digit or self._codec.semantic_vocab_size != codec.semantic_vocab_size:
                raise ValueError("TIGERModel was already initialized with an incompatible SID codec.")
            return

        self._codec = codec
        if self.config.path_ranking_enabled:
            buckets = defaultdict(list)
            for tokens in codec.item_to_tokens.values():
                buckets[tuple(tokens[:2])].append(tuple(tokens))
            self._ranking_buckets = dict(buckets)
        branches = [defaultdict(set) for _ in range(codec.n_digit)]
        for tokens in codec.item_to_tokens.values():
            for depth, child in enumerate(tokens):
                branches[depth][tuple(int(value) for value in tokens[:depth])].add(int(child))
        self._parent_prefix_children = tuple(
            {parent: tuple(sorted(children)) for parent, children in depth_branches.items()}
            for depth_branches in branches
        )
        self._base_user_token = codec.semantic_vocab_size + 1
        self._eos_token = self._base_user_token + int(self.config.n_user_tokens)
        self._vocab_size = self._eos_token + 1
        self._max_observed_history_items = _max_observed_history_items(prepared_data)
        signatures = _load_user_signatures(
            self.config.user_signature_file,
            min_items=int(self.config.user_signature_min_items),
            semantic_vocab_size=int(codec.semantic_vocab_size),
            disjoint=bool(self.config.user_signature_disjoint),
            base_user_token=int(self._base_user_token),
            n_user_tokens=int(self.config.n_user_tokens),
        )
        self._max_signature_tokens = max((len(tokens) for tokens in signatures.values()), default=0)
        self._soft_prompt = self._build_soft_prompt(prepared_data)
        self._soft_prompt_tokens = 0 if self._soft_prompt is None else self._soft_prompt.n_tokens
        self._align_head = self._build_align_head(prepared_data)
        self._t5 = self._build_t5()
        self._inject_text_embeddings()
        self._build_literal_text_encoder()

    def _build_literal_text_encoder(self) -> None:
        if not str(getattr(self.config, "literal_text_file", "") or "").strip():
            return
        if str(getattr(self.config, "text_embed_mode", "off") or "off").lower() != "off":
            raise ValueError("literal_text_file and text_embed_mode are mutually exclusive C1 variants.")
        from transformers import T5EncoderModel

        pretrained = T5EncoderModel.from_pretrained(self.config.literal_text_model)
        embedding = pretrained.get_input_embeddings()
        embedding.weight.requires_grad_(bool(self.config.literal_text_finetune))
        self._literal_text_embedding = embedding
        self._literal_text_projection = torch.nn.Linear(
            int(embedding.embedding_dim), int(self.config.d_model), bias=False
        )

    def _inject_text_embeddings(self) -> None:
        """TRACE C1-text: text-ground the SID token embeddings from pretrained phrase vectors.

        Loads a ``sid_text_embed.pt`` ({embed [V,d_text], has_text [V]}), PCA-projects each
        phrase vector to d_model (deterministic; preserves phrase-similarity structure so codes
        sharing words start with similar embeddings), scales to the T5 init std, and writes the
        SID token rows of the tied embedding. ``finetune`` trains from this init; ``frozen``
        keeps phrase-backed rows fixed; ``off`` no-ops.
        """
        mode = str(getattr(self.config, "text_embed_mode", "off") or "off").lower()
        path = str(getattr(self.config, "text_embed_file", "") or "").strip()
        depth_spec = str(self.config.text_embed_depths).strip().lower()
        if depth_spec != "all" and (mode == "off" or not path):
            raise ValueError("text_embed_depths requires enabled text embeddings and text_embed_file.")
        if mode == "off" or not path:
            return
        if mode not in ("finetune", "frozen"):
            raise ValueError(f"text_embed_mode must be off|finetune|frozen, got {mode!r}")
        anchor_weight = float(getattr(self.config, "text_anchor_weight", 0.0))
        if anchor_weight < 0:
            raise ValueError("text_anchor_weight must be non-negative.")
        if anchor_weight > 0 and mode != "finetune":
            raise ValueError("text_anchor_weight requires text_embed_mode=finetune.")
        data = torch.load(path, map_location="cpu", weights_only=False)
        embed = data["embed"].float()          # [V, d_text]
        has_text = data["has_text"].bool()     # [V]
        emb_layer = self._t5.get_input_embeddings()
        vocab_size = int(emb_layer.weight.shape[0])
        d_model = int(self.config.d_model)

        # Select rows without changing the PCA basis, scaling, or random stream.
        # This makes selected rows identical to full Lexical, and others to Atomic.
        inject_mask = has_text.clone()
        if depth_spec != "all":
            codec = self._codec
            if codec is None:
                raise RuntimeError("Depth-specific lexical initialization requires a SID codec.")
            try:
                depths = {int(value) for value in depth_spec.replace(",", ":").split(":")}
            except ValueError as exc:
                raise ValueError("text_embed_depths must be 'all' or colon-separated 1-based depths.") from exc
            if not depths or min(depths) < 1 or max(depths) > codec.n_digit:
                raise ValueError(f"text_embed_depths must be within 1..{codec.n_digit}.")
            token_depth = {}
            inject_mask.zero_()
            for tokens in codec.item_to_tokens.values():
                for depth, token in enumerate(tokens, start=1):
                    if token_depth.setdefault(token, depth) != depth:
                        raise ValueError("Depth-specific lexical initialization requires depth-unique token rows.")
                    if depth in depths and token < len(has_text) and bool(has_text[token]):
                        inject_mask[token] = True
            if not bool(inject_mask[:vocab_size].any()):
                raise ValueError("Selected lexical depths contain no phrase-backed SID rows.")

        rows = embed[has_text]
        if rows.numel() == 0:
            if self.config.phrase_loss_enabled:
                raise ValueError("Phrase loss requires phrase-backed rows")
            return
        mean = rows.mean(dim=0, keepdim=True)
        # PCA via SVD of the centered text vectors -> top directions (up to d_model).
        _, _, vt = torch.linalg.svd(rows - mean, full_matrices=False)
        comps = vt[:d_model]                    # [k<=d_model, d_text]
        proj = (embed - mean) @ comps.t()       # [V, k]
        if proj.shape[1] < d_model:             # pad low-rank / tiny-vocab cases to d_model
            proj = torch.cat([proj, proj.new_zeros(proj.shape[0], d_model - proj.shape[1])], dim=1)
        with torch.no_grad():
            tgt_std = float(emb_layer.weight.std().item())
            cur_std = float(proj[has_text].std().item()) or 1.0
            proj = proj * (tgt_std / cur_std)
            n_inject = 0
            upto = min(int(embed.shape[0]), vocab_size)
            for tok in range(upto):
                if bool(inject_mask[tok]):
                    emb_layer.weight[tok] = proj[tok].to(emb_layer.weight.dtype)
                    n_inject += 1
        if self.config.phrase_loss_enabled:
            device = emb_layer.weight.device
            self._phrase_targets = torch.zeros(vocab_size, embed.shape[1], device=device)
            self._phrase_targets[:upto] = embed[:upto].to(device)
            self._phrase_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            # Only native SID token rows with nonzero phrase vectors are eligible.
            for tokens in self._codec.item_to_tokens.values():
                for token in tokens:
                    if token < upto and bool(has_text[token]) and embed[token].norm() > 0:
                        self._phrase_mask[token] = True
            if not self._phrase_mask.any():
                raise ValueError("Phrase loss requires nonzero phrase-backed SID rows")
            self._phrase_projection = nn.Linear(d_model, embed.shape[1], bias=False).to(device)
        if anchor_weight > 0:
            depth_weights = self._text_anchor_depth_weights()
            token_depth: dict[int, int] = {}
            codec = self._codec
            if codec is None:
                raise RuntimeError("TIGER lexical anchor requires an initialized SID codec.")
            for tokens in codec.item_to_tokens.values():
                for depth, token in enumerate(tokens):
                    previous = token_depth.setdefault(int(token), depth)
                    if previous != depth:
                        raise ValueError(
                            "Lexical anchor requires partition-preserving depth-unique SID tokens; "
                            f"token {token} appears at depths {previous} and {depth}."
                        )
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=emb_layer.weight.device)
            row_weights = torch.zeros(vocab_size, dtype=torch.float32, device=emb_layer.weight.device)
            for token, depth in token_depth.items():
                if token < upto and bool(inject_mask[token]) and depth_weights[depth] > 0:
                    mask[token] = True
                    row_weights[token] = depth_weights[depth]
            if not bool(mask.any()):
                raise ValueError("text_anchor_weight is enabled but no phrase-backed SID rows were selected.")
            self._text_anchor_row_mask = mask
            anchor_values = torch.zeros_like(emb_layer.weight)
            anchor_values[:upto] = proj[:upto].to(device=anchor_values.device, dtype=anchor_values.dtype)
            self._text_anchor_row_values = anchor_values
            self._text_anchor_row_weights = row_weights
        if mode == "frozen":
            output_layer = self._t5.get_output_embeddings()
            if output_layer is not None and output_layer.weight is not emb_layer.weight:
                raise RuntimeError("Frozen lexical C1 requires tied T5 input/output embeddings.")
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=emb_layer.weight.device)
            mask[:upto] = inject_mask[:upto].to(device=mask.device)
            self._frozen_text_row_mask = mask
            self._frozen_text_row_values = emb_layer.weight.detach()[mask].clone()
            if self._frozen_text_grad_hook is not None:
                self._frozen_text_grad_hook.remove()
            self._frozen_text_grad_hook = emb_layer.weight.register_hook(
                lambda grad: grad.masked_fill(self._frozen_text_row_mask[:, None], 0)
            )
        print(
            f"[c1-text] text-grounded {n_inject} SID token embeddings from {os.path.basename(path)} "
            f"(mode={mode}, depths={depth_spec}, d_text={int(embed.shape[1])}->d_model={d_model})",
            flush=True,
        )

    def _text_anchor_depth_weights(self) -> list[float]:
        """Return one non-negative lexical-anchor multiplier per SID depth."""
        codec = self._codec
        if codec is None:
            raise RuntimeError("TIGER lexical anchor requires an initialized SID codec.")
        raw = str(getattr(self.config, "text_anchor_depth_weights", "") or "").strip()
        if not raw:
            return [1.0] * int(codec.n_digit)
        try:
            weights = [float(value.strip()) for value in raw.replace(":", ",").split(",")]
        except ValueError as exc:
            raise ValueError("text_anchor_depth_weights must be comma- or colon-separated floats.") from exc
        if len(weights) != int(codec.n_digit) or any(weight < 0 for weight in weights):
            raise ValueError(
                f"text_anchor_depth_weights must contain {codec.n_digit} non-negative values, got {raw!r}."
            )
        return weights

    def _text_anchor_loss(self) -> torch.Tensor | None:
        """Mean depth-weighted squared drift from the projected phrase initialization."""
        mask = self._text_anchor_row_mask
        targets = self._text_anchor_row_values
        weights = self._text_anchor_row_weights
        if mask is None or targets is None or weights is None:
            return None
        current = self._t5_module().get_input_embeddings().weight
        selected_weights = weights[mask].to(device=current.device, dtype=current.dtype)
        squared = (current[mask] - targets[mask].to(device=current.device, dtype=current.dtype)).square().mean(dim=1)
        return (squared * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)

    def _parent_prefix_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Gold-child cross entropy normalized only over valid sibling branches.

        Standard language-model CE competes against the complete vocabulary.
        This auxiliary term explicitly trains the local decision made by SID
        decoding: select the gold child among children of the current gold
        parent prefix. Deterministic one-child branches contribute no loss.
        """
        codec = self._sid_codec()
        if labels.ndim != 2 or logits.ndim != 3:
            raise ValueError("Parent-prefix supervision expects [B,L] labels and [B,L,V] logits.")
        if labels.shape[1] < codec.n_digit or logits.shape[1] < codec.n_digit:
            raise ValueError("Training outputs are shorter than the SID width.")

        total = logits.new_zeros(())
        decisions = 0
        labels_cpu = labels[:, : codec.n_digit].detach().cpu()
        vocab_size = int(logits.shape[-1])
        for depth, branch_map in enumerate(self._parent_prefix_children):
            allowed = torch.zeros(
                (labels_cpu.shape[0], vocab_size), dtype=torch.bool
            )
            ambiguous = torch.zeros(labels_cpu.shape[0], dtype=torch.bool)
            for row in range(labels_cpu.shape[0]):
                parent = tuple(int(value) for value in labels_cpu[row, :depth].tolist())
                children = branch_map.get(parent)
                if children is None:
                    raise ValueError(f"Gold parent prefix is absent from the SID tree: {parent}")
                if len(children) <= 1:
                    continue
                allowed[row, list(children)] = True
                ambiguous[row] = True
            if not bool(ambiguous.any()):
                continue
            allowed = allowed.to(logits.device)
            ambiguous = ambiguous.to(logits.device)
            local_logits = logits[:, depth].masked_fill(~allowed, -torch.inf)
            total = total + nn.functional.cross_entropy(
                local_logits[ambiguous], labels[ambiguous, depth].long(), reduction="sum"
            )
            decisions += int(ambiguous.sum().item())
        if decisions == 0:
            return logits.sum() * 0.0
        return total / decisions

    def _path_ranking_loss(self, batch, outputs, encoder_mask):
        """Sampled full-path CE: half shared-prefix and half catalog negatives.

        Every negative is teacher-forced under its own prefix, including suffix and EOS.
        Reuse encoder states to avoid encoding each history repeatedly. Sampling uses
        training history and catalog SIDs only; no validation/test interactions are read.
        """
        from transformers.modeling_outputs import BaseModelOutput

        codec = self._sid_codec()
        labels = batch["labels"]
        width = codec.n_digit
        count = int(self.config.path_ranking_negatives)
        catalog = list(codec.tokens_to_item)
        negatives = []
        for row, gold_row in enumerate(labels.detach().cpu().tolist()):
            gold = tuple(gold_row[:width])
            history = batch["input_ids"][row].detach().cpu().tolist()
            excluded = {gold}
            if "ranking_seen_item_ids" in batch:
                excluded.update(
                    codec.item_tokens(item_id)
                    for item_id in batch["ranking_seen_item_ids"][row].detach().cpu().tolist()
                    if item_id >= 0
                )
            # Sliding lookup also supports user/signature tokens preceding history.
            for start in range(len(history) - width + 1):
                path = tuple(history[start:start + width])
                if path in codec.tokens_to_item:
                    excluded.add(path)
            global_pool = [path for path in catalog if path not in excluded]
            if not global_pool:
                raise ValueError("No unseen negative items available for path ranking")
            local_pool = [path for path in self._ranking_buckets.get(gold[:2], []) if path not in excluded]
            chosen = []
            for index in range(count):
                pool = local_pool if index < count // 2 and local_pool else global_pool
                pool = [path for path in pool if path not in chosen] or global_pool
                path = pool[int(torch.randint(len(pool), ()).item())]
                chosen.append(path)
                negatives.append(list(path) + [self._eos_token])
        negative_labels = labels.new_tensor(negatives)
        enc = outputs.encoder_last_hidden_state.repeat_interleave(count, dim=0)
        mask = encoder_mask.repeat_interleave(count, dim=0)
        negative_outputs = self._t5_module()(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc),
            attention_mask=mask, labels=negative_labels, use_cache=False,
        )
        positive_score = self._sequence_log_score(outputs.logits, labels)
        negative_score = self._sequence_log_score(negative_outputs.logits, negative_labels).reshape(-1, count)
        scores = torch.cat([positive_score[:, None], negative_score], dim=1)
        return nn.functional.cross_entropy(
            scores / float(self.config.path_ranking_temperature),
            torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long),
        )

    def _prefix_survival_depths(self) -> tuple[int, ...]:
        """Validated 1-based depths receiving beam-aware survival supervision."""
        weight = float(self.config.prefix_survival_weight)
        if weight < 0:
            raise ValueError("prefix_survival_weight must be non-negative.")
        if weight == 0:
            return ()
        codec = self._sid_codec()
        raw = str(self.config.prefix_survival_depths or "").strip()
        if not raw:
            raise ValueError("prefix_survival_depths is required when prefix_survival_weight > 0.")
        try:
            depths = tuple(sorted({int(value.strip()) for value in raw.replace(":", ",").split(",")}))
        except ValueError as exc:
            raise ValueError("prefix_survival_depths must be comma- or colon-separated integers.") from exc
        if not depths or min(depths) < 1 or max(depths) >= int(codec.n_digit):
            raise ValueError(
                f"prefix_survival_depths must be within 1..{int(codec.n_digit) - 1}; got {raw!r}."
            )
        if float(self.config.prefix_survival_margin) < 0:
            raise ValueError("prefix_survival_margin must be non-negative.")
        if int(self.config.prefix_survival_num_beams) < 2:
            raise ValueError("prefix_survival_num_beams must be at least 2.")
        if int(self.config.prefix_survival_examples) < 0:
            raise ValueError("prefix_survival_examples must be non-negative.")
        return depths

    @staticmethod
    def _slice_encoder_kwargs(
        encoder_kwargs: Mapping[str, torch.Tensor], count: int
    ) -> dict[str, torch.Tensor]:
        return {key: value[:count] for key, value in encoder_kwargs.items()}

    @staticmethod
    def _sequence_log_score(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Unnormalized cumulative token log probability for equal-length prefixes."""
        token_log_prob = logits.float().log_softmax(dim=-1)
        return token_log_prob.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    def _prefix_survival_loss(
        self,
        *,
        encoder_kwargs: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        positive_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Margin loss against beam-cutoff hard prefixes mined from the current decoder.

        Hard-negative beam search is discrete and runs without gradients. The selected
        cutoff prefix is then teacher-forced through the same decoder, making its score and
        the gold-prefix score differentiable. Rows where the cutoff equals the gold prefix
        are ignored because they provide no ranking contrast.
        """
        depths = self._prefix_survival_depths()
        if not depths:
            return positive_logits.new_zeros(())
        local_batch = int(labels.shape[0])
        configured = int(self.config.prefix_survival_examples)
        count = local_batch if configured == 0 else min(local_batch, configured)
        if count <= 0:
            return positive_logits.new_zeros(())
        beam_width = int(self.config.prefix_survival_num_beams)
        margin = float(self.config.prefix_survival_margin)
        t5 = self._t5_module()
        mined_encoder = self._slice_encoder_kwargs(encoder_kwargs, count)
        losses: list[torch.Tensor] = []

        for depth in depths:
            was_training = bool(t5.training)
            try:
                t5.eval()
                with torch.no_grad():
                    generated = t5.generate(
                        max_new_tokens=int(depth),
                        min_new_tokens=int(depth),
                        num_beams=beam_width,
                        num_return_sequences=beam_width,
                        return_dict_in_generate=True,
                        output_scores=True,
                        use_cache=True,
                        **mined_encoder,
                    )
            finally:
                t5.train(was_training)

            sequences = generated.sequences.reshape(count, beam_width, -1)
            sequence_scores = generated.sequences_scores.reshape(count, beam_width)
            cutoff_index = sequence_scores.argmin(dim=1)
            row_index = torch.arange(count, device=sequences.device)
            negative = sequences[row_index, cutoff_index, 1 : 1 + int(depth)].long()
            positive = labels[:count, : int(depth)].long()
            useful = (negative != positive).any(dim=1)
            if not bool(useful.any()):
                continue

            negative_outputs = t5(labels=negative, **mined_encoder)
            positive_score = self._sequence_log_score(
                positive_logits[:count, : int(depth)], positive
            )
            negative_score = self._sequence_log_score(negative_outputs.logits, negative)
            losses.append(
                nn.functional.relu(margin + negative_score[useful] - positive_score[useful]).mean()
            )

        if not losses:
            return positive_logits.new_zeros(())
        return torch.stack(losses).mean()

    def restore_frozen_text_embedding_rows(self) -> None:
        """Undo optimizer weight decay or momentum updates on frozen lexical SID rows."""
        mask = self._frozen_text_row_mask
        values = self._frozen_text_row_values
        if mask is None or values is None:
            return
        emb_layer = self._t5_module().get_input_embeddings()
        with torch.no_grad():
            emb_layer.weight[mask] = values.to(device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)

    def _build_soft_prompt(self, prepared_data: TIGERModelDataset) -> "_CFSoftPrompt | None":
        path = str(self.config.cf_soft_prompt_file or "").strip()
        if not path:
            return None
        profile_path = Path(path)
        if not profile_path.exists():
            raise FileNotFoundError(f"TIGER cf_soft_prompt_file does not exist: {profile_path}")
        payload = torch.load(profile_path, map_location="cpu")
        raw_user_ids = payload["user_ids"].long()
        raw_profiles = payload["profiles"].float()
        n_train = payload["n_train"].long()
        d_cf = int(raw_profiles.shape[1])
        num_users = int(prepared_data.get_num_users())
        # Cold/missing users keep the zero (== mean, post-standardization) profile so the
        # MLP maps them to a single learned "average user" prefix.
        min_items = int(self.config.cf_soft_prompt_min_items)
        profiles = torch.zeros((num_users, d_cf), dtype=torch.float32)
        eligible = 0
        for row in range(int(raw_user_ids.shape[0])):
            user_id = int(raw_user_ids[row])
            if user_id < 0 or user_id >= num_users:
                continue
            if int(n_train[row]) < min_items:
                continue
            profiles[user_id] = raw_profiles[row]
            eligible += 1
        if eligible == 0:
            raise ValueError(
                f"cf_soft_prompt_file {profile_path} yielded 0 eligible users for "
                f"num_users={num_users}, min_items={min_items}. Check id-space alignment."
            )
        return _CFSoftPrompt(
            profiles,
            d_model=int(self.config.d_model),
            n_tokens=int(self.config.cf_soft_prompt_tokens),
            hidden=int(self.config.cf_soft_prompt_hidden),
            dropout=float(self.config.cf_soft_prompt_dropout),
        )

    def _build_align_head(self, prepared_data: TIGERModelDataset) -> "_CFAlignHead | None":
        path = str(self.config.cf_align_file or "").strip()
        if not path or float(self.config.cf_align_weight) == 0.0:
            return None
        profile_path = Path(path)
        if not profile_path.exists():
            raise FileNotFoundError(f"TIGER cf_align_file does not exist: {profile_path}")
        payload = torch.load(profile_path, map_location="cpu")
        raw_user_ids = payload["user_ids"].long()
        raw_profiles = payload["profiles"].float()
        n_train = payload["n_train"].long()
        d_cf = int(raw_profiles.shape[1])
        num_users = int(prepared_data.get_num_users())
        min_items = int(self.config.cf_align_min_items)
        profiles = torch.zeros((num_users, d_cf), dtype=torch.float32)
        eligible = torch.zeros((num_users,), dtype=torch.bool)
        n_eligible = 0
        for row in range(int(raw_user_ids.shape[0])):
            user_id = int(raw_user_ids[row])
            if user_id < 0 or user_id >= num_users:
                continue
            if int(n_train[row]) < min_items:
                continue
            profiles[user_id] = raw_profiles[row]
            eligible[user_id] = True
            n_eligible += 1
        if n_eligible == 0:
            raise ValueError(
                f"cf_align_file {profile_path} yielded 0 eligible users for "
                f"num_users={num_users}, min_items={min_items}. Check id-space alignment."
            )
        return _CFAlignHead(
            profiles,
            eligible,
            d_model=int(self.config.d_model),
            hidden=int(self.config.cf_align_hidden),
            temp=float(self.config.cf_align_temp),
        )

    def _encoder_kwargs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_ids: torch.Tensor | None,
        literal_input_ids: torch.Tensor | None = None,
        literal_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encoder inputs: plain token ids, or token embeds with a prepended CF soft prompt."""
        if literal_input_ids is None and self._soft_prompt is None:
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        t5 = self._t5_module()
        if literal_input_ids is not None:
            if user_ids is None or literal_attention_mask is None:
                raise ValueError("Literal-text encoder requires user_ids and literal_attention_mask.")
            if self._literal_text_embedding is None or self._literal_text_projection is None:
                raise RuntimeError("Literal-text encoder modules were not initialized.")
            text_embeds = self._literal_text_projection(self._literal_text_embedding(literal_input_ids))
            user_tokens = self._require_base_user_token() + user_ids.remainder(int(self.config.n_user_tokens))
            user_embeds = t5.get_input_embeddings()(user_tokens[:, None])
            inputs_embeds = torch.cat([user_embeds, text_embeds.to(user_embeds.dtype)], dim=1)
            full_mask = torch.cat([
                literal_attention_mask.new_ones((literal_attention_mask.shape[0], 1)),
                literal_attention_mask,
            ], dim=1)
        else:
            inputs_embeds = t5.get_input_embeddings()(input_ids)
            full_mask = attention_mask
        if self._soft_prompt is not None:
            if user_ids is None:
                raise ValueError("CF soft prompt is enabled but the batch is missing 'user_ids'.")
            prefix = self._soft_prompt(user_ids).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
            prefix_mask = full_mask.new_ones((full_mask.shape[0], prefix.shape[1]))
            full_mask = torch.cat([prefix_mask, full_mask], dim=1)
        return {"inputs_embeds": inputs_embeds, "attention_mask": full_mask}

    def _require_base_user_token(self) -> int:
        if self._base_user_token is None:
            raise RuntimeError("TIGERModel user-token range is not initialized.")
        return self._base_user_token

    def _build_t5(self) -> torch.nn.Module:
        try:
            from transformers import T5Config, T5ForConditionalGeneration
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("TIGER requires `transformers`. Install it before using model.name=tiger.") from exc

        max_positions = self._max_positions()
        t5_config = T5Config(
            num_layers=self.config.num_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            num_heads=self.config.num_heads,
            d_kv=self.config.d_kv,
            dropout_rate=self.config.dropout_rate,
            activation_function=self.config.activation_function,
            vocab_size=self._require_vocab_size(),
            pad_token_id=self._pad_token,
            eos_token_id=self._require_eos_token(),
            decoder_start_token_id=self._pad_token,
            feed_forward_proj=self.config.feed_forward_proj,
            n_positions=max_positions,
        )
        return T5ForConditionalGeneration(config=t5_config)

    def _max_positions(self) -> int:
        history_max_length = self.config.history_max_length
        if history_max_length is None:
            # None means unbounded history in SequentialModelConfig. The model dataset
            # has already materialized histories, so use the observed maximum instead
            # of collapsing to an empty-history T5 position budget.
            max_history_items = max(int(self._max_observed_history_items or 0), 1)
        else:
            max_history_items = int(history_max_length)
        prefix_budget = max(1, int(self._max_signature_tokens or 0))
        return (
            max_history_items * self._sid_codec().n_digit
            + prefix_budget
            + int(self._soft_prompt_tokens or 0)
            + 1
        )

    def _t5_module(self) -> torch.nn.Module:
        if self._t5 is None:
            raise RuntimeError("TIGERModel must be initialized via build_train_collator/build_eval_collator before use.")
        return self._t5

    def _sid_codec(self) -> TIGERSIDCodec:
        if self._codec is None:
            raise RuntimeError("TIGERModel has no SID codec. Build a collator with prepared data first.")
        return self._codec

    def _require_eos_token(self) -> int:
        if self._eos_token is None:
            raise RuntimeError("TIGERModel eos token is not initialized.")
        return self._eos_token

    def _require_vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("TIGERModel vocab size is not initialized.")
        return self._vocab_size

    @staticmethod
    def _excluded_item_sets(
        exclude_item_ids: torch.Tensor | None,
        exclude_mask: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> list[set[int]]:
        excluded = [set() for _ in range(batch_size)]
        if exclude_item_ids is None or exclude_mask is None or exclude_item_ids.numel() == 0:
            return excluded
        ids = exclude_item_ids.detach().cpu()
        mask = exclude_mask.detach().cpu().to(dtype=torch.bool)
        for row_index in range(min(batch_size, int(ids.shape[0]))):
            excluded[row_index] = {
                int(item_id)
                for item_id, keep in zip(ids[row_index].tolist(), mask[row_index].tolist(), strict=False)
                if keep
            }
        return excluded


def _max_observed_history_items(prepared_data: TIGERModelDataset) -> int:
    max_history = 0
    for split in ("train", "valid", "test"):
        dataset = prepared_data.get_train_dataset() if split == "train" else prepared_data.get_eval_dataset(split)
        frame = getattr(dataset, "frame", None)
        if frame is None or HISTORY_ITEM_IDS not in frame:
            continue
        for history in frame[HISTORY_ITEM_IDS].tolist():
            max_history = max(max_history, len(history or ()))
    return max_history


__all__ = ["TIGERModel"]
