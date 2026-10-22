from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class TIGERConfig(SequentialModelConfig):
    """Configuration for the TIGER generative retrieval model."""

    name: str = field(default="tiger", metadata={"help": "Registered model name."})
    sid_file: str = field(default="", metadata={"help": "Path to item_sids.json using remapped item ids as keys."})
    history_max_length: int | None = field(
        default=20,
        metadata={"help": "Maximum number of recent history items used by TIGER."},
    )
    n_user_tokens: int = field(default=1, metadata={"help": "Number of hash buckets used for user tokens."})
    user_signature_file: str = field(
        default="",
        metadata={
            "help": (
                "Optional offline user_sid_medoids JSON. Eligible users (n_train_unique_items >= "
                "user_signature_min_items) prepend signature_codes; others keep the hashed user token."
            )
        },
    )
    user_signature_min_items: int = field(
        default=5,
        metadata={"help": "Minimum n_train_unique_items required to use a medoid user signature."},
    )
    user_signature_disjoint: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, signature_codes are mapped into the disjoint user-token band "
                "(code -> base_user_token + code, validated 0 <= code < n_user_tokens) instead of "
                "the item token range. Use for CF-residual user SIDs; set n_user_tokens >= "
                "the builder's required_n_user_tokens (levels * codebook_size)."
            )
        },
    )
    cf_soft_prompt_file: str = field(
        default="",
        metadata={
            "help": (
                "Optional .pt of per-user continuous CF profiles (build_user_cf_profile.py). When set, "
                "the standardized profile e_u is projected through a small learned MLP and prepended as "
                "cf_soft_prompt_tokens continuous prefix embedding(s) on the T5 encoder (P-Tuning style). "
                "No vocabulary, no quantization, no token collision -- Option 3 ceiling ablation."
            )
        },
    )
    cf_soft_prompt_tokens: int = field(
        default=1,
        metadata={"help": "Number of continuous CF prefix positions prepended to the encoder input."},
    )
    cf_soft_prompt_hidden: int = field(
        default=128,
        metadata={"help": "Hidden width of the CF profile -> prefix-embedding MLP."},
    )
    cf_soft_prompt_min_items: int = field(
        default=5,
        metadata={"help": "Users with fewer unique train items fall back to the zero (mean) CF profile."},
    )
    cf_soft_prompt_dropout: float = field(
        default=0.0,
        metadata={"help": "Feature dropout applied to the CF profile before the MLP at train time."},
    )
    cf_align_file: str = field(
        default="",
        metadata={
            "help": (
                "Optional .pt of per-user continuous CF profiles (build_user_cf_profile.py, same "
                "format as cf_soft_prompt_file). When set, an auxiliary LETTER-style contrastive "
                "loss aligns the encoder's mean-pooled history representation to the user's CF "
                "profile e_u. NO token is added (unlike the prefix schemes): CF enters only through "
                "the loss, so history_max_length and the SID vocabulary are untouched."
            )
        },
    )
    cf_align_weight: float = field(
        default=0.1,
        metadata={"help": "Weight of the auxiliary CF alignment loss added to the LM loss (0 disables)."},
    )
    cf_align_hidden: int = field(
        default=128,
        metadata={"help": "Hidden width of the pooled-encoder -> CF-space projection MLP."},
    )
    cf_align_min_items: int = field(
        default=5,
        metadata={"help": "Users with fewer unique train items are excluded from the alignment loss."},
    )
    cf_align_temp: float = field(
        default=0.1,
        metadata={"help": "Temperature for the InfoNCE alignment (cosine similarities / temp)."},
    )
    adaptive_depth: bool = field(
        default=False,
        metadata={
            "help": (
                "Inference-time adaptive-depth decoding (efficiency C3). When True, predict() stops "
                "each query at the shallowest SID prefix depth d<=L whose top-K item set matches the "
                "full-depth top-K within tolerance adaptive_tau, resolving the depth-d prefix bucket "
                "by the top beam in it. Frozen model; off by default (== fixed full-depth baseline)."
            )
        },
    )
    adaptive_tau: float = field(
        default=0.0,
        metadata={
            "help": (
                "Necessity tolerance for adaptive_depth. Stop at depth d when 1 - Jaccard(topK@d, "
                "topK@L) <= tau. tau=0 requires an exact top-K set match to stop early; larger tau "
                "trades a bit of top-K change for shallower (cheaper) decoding."
            )
        },
    )
    adaptive_min_depth: int = field(
        default=1,
        metadata={"help": "Smallest prefix depth adaptive_depth is allowed to stop at."},
    )
    adaptive_stop: str = field(
        default="oracle",
        metadata={
            "help": (
                "How the stop depth is decided. 'oracle' (default) compares topK@d to the full-depth "
                "topK@L (1 - Jaccard <= adaptive_tau) -- the retrospective upper bound; it needs the "
                "full decode and cannot save wall-clock. 'running' is the deployable, peek-free rule: "
                "stop at the shallowest d where topK@d has been stable vs the previous depth "
                "(1 - Jaccard(topK@d, topK@(d-1)) <= adaptive_tau) for adaptive_stable_m consecutive "
                "depths -- never references L, so the decoder can truly halt early."
            )
        },
    )
    adaptive_stable_m: int = field(
        default=1,
        metadata={
            "help": (
                "Running-mode only: number of consecutive depths the top-K set must stay stable "
                "before stopping. m=1 stops as soon as topK@d == topK@(d-1); larger m is more "
                "conservative (later exit, safer)."
            )
        },
    )
    adaptive_metric: str = field(
        default="jaccard",
        metadata={
            "help": (
                "Necessity metric for adaptive-depth. 'jaccard' (default) = set overlap "
                "(order-insensitive, recall/hit-ratio flavored). 'ndcg' = order-AWARE consistency: "
                "1 - DCG(topK@d ordering, scored by the reference ranks) / DCG(reference); the "
                "reference is topK@L (oracle) or topK@(d-1) (running). tau=0 with 'ndcg' stops only "
                "once the RANKING (not just the set) is settled, protecting NDCG at tau>0 and in the "
                "running rule."
            )
        },
    )
    adaptive_inloop: bool = field(
        default=False,
        metadata={
            "help": (
                "Running mode only: actually halt generate() at the settled depth (banks wall-clock), "
                "instead of the retrospective post-hoc wrapper. Requires adaptive_depth=True and "
                "adaptive_stop='running'. Item names for a committed depth-d prefix come from a static "
                "prefix->item table built offline from the frozen SID assignment (a free lookup, not a "
                "runtime full-depth decode). Off by default; enabling it only changes the running path."
            )
        },
    )
    text_embed_file: str = field(
        default="",
        metadata={"help": (
            "TRACE C1-text: path to a sid_text_embed.pt (from scripts/build_sid_text_embed.py) that "
            "text-grounds SID token embeddings. Requires position-unique SIDs (map_sids_to_words "
            "--offset_columns). Empty == current atomic C1.")},
    )
    text_embed_mode: str = field(
        default="off",
        metadata={"help": (
            "How text_embed_file is used. 'off' (default) = unchanged atomic tokens. 'finetune' = "
            "initialize each SID token's embedding from a PCA-projection of its phrase's sentence-T5 "
            "vector, then train freely. 'frozen' = keep SID rows fixed to the projected text (ablation).")},
    )
    text_embed_depths: str = field(
        default="all",
        metadata={"help": (
            "Depths receiving lexical initialization: 'all' or colon-separated 1-based "
            "positions (e.g. '1:2'). Other rows retain random initialization. PCA and "
            "scaling always use all phrase vectors, matching full Lexical C1."
        )},
    )
    text_anchor_weight: float = field(
        default=0.0,
        metadata={"help": (
            "TRACE C1 lexical-anchor weight. When > 0, add a depth-weighted MSE that keeps "
            "phrase-backed SID token rows near their text-grounded initialization. Requires "
            "text_embed_mode=finetune; 0 disables the auxiliary loss."
        )},
    )
    text_anchor_depth_weights: str = field(
        default="",
        metadata={"help": (
            "Comma- or colon-separated non-negative anchor multipliers, one per SID depth. Empty "
            "means 1 for every depth. Example '1,1,0,0' anchors only the first two of four codes."
        )},
    )
    path_ranking_enabled: bool = False
    path_ranking_weight_mode: str = "learned"
    path_ranking_weight: float = 0.1
    path_ranking_negatives: int = 4
    path_ranking_temperature: float = 1.0
    phrase_loss_enabled: bool = False
    phrase_loss_weight_mode: str = "learned"
    phrase_loss_weight: float = 0.1
    parent_prefix_weight_mode: str = "fixed"
    parent_prefix_loss_weight: float = field(
        default=0.0,
        metadata={"help": (
            "Weight of the parent-prefix branch auxiliary loss. At each teacher-forced SID "
            "position, normalize logits only over valid sibling children of the gold parent "
            "prefix and classify the gold child. Zero disables the loss."
        )},
    )
    prefix_survival_weight: float = field(
        default=0.0,
        metadata={"help": (
            "Weight of the beam-aware target-prefix survival loss. The loss mines the "
            "beam-cutoff prefix without gradients, then differentiably scores it against "
            "the gold prefix. Zero disables the experiment."
        )},
    )
    prefix_survival_depths: str = field(
        default="",
        metadata={"help": (
            "Comma- or colon-separated 1-based SID depths receiving beam-survival "
            "supervision. Required when prefix_survival_weight > 0; full depth is rejected."
        )},
    )
    prefix_survival_margin: float = field(
        default=0.1,
        metadata={"help": "Margin between the gold-prefix score and beam-cutoff hard negative."},
    )
    prefix_survival_num_beams: int = field(
        default=40,
        metadata={"help": "Beam width used only to mine prefix-survival hard negatives during training."},
    )
    prefix_survival_examples: int = field(
        default=16,
        metadata={"help": (
            "Maximum examples per local training batch receiving the expensive survival "
            "loss. Zero means the full local batch."
        )},
    )
    literal_text_file: str = field(
        default="",
        metadata={"help": (
            "Encoder-only literal-text C1: token_text.json mapping SID token ids to phrases. "
            "When set, history items are tokenized with literal_text_model and fed as pretrained "
            "text-token embeddings; decoder labels remain the original discrete SID tuples.")},
    )
    literal_text_model: str = field(
        default="sentence-transformers/sentence-t5-base",
        metadata={"help": "Local Hugging Face T5 tokenizer/embedding model for literal-text histories."},
    )
    literal_text_max_tokens: int = field(
        default=512,
        metadata={"help": "Maximum literal text tokens per history, retaining the most recent tokens."},
    )
    literal_text_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune pretrained literal-text token embeddings (default: freeze them)."},
    )
    recode_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "C4 (codebook redistribution), minimal/model-free variant. When True, the loaded SID "
                "is recoded BEFORE tokenization: at the target position(s), items are reassigned "
                "across the codeword values already present in each parent-prefix group to balance "
                "code utilization (max entropy), preserving the parent prefix + suffix and item<->tuple "
                "uniqueness. Repairs collapsed/degenerate SIDs; off by default (== identity)."
            )
        },
    )
    recode_positions: str = field(
        default="auto",
        metadata={
            "help": (
                "recode_enabled only. 'auto' = the single position with the largest achievable "
                "utilization-entropy gain (constant and binary/dedup positions skipped); or a "
                "comma-separated list of 0-indexed positions applied in order."
            )
        },
    )
    recode_signal_file: str = field(
        default="",
        metadata={
            "help": (
                "recode_enabled only. Path to a [num_items, d] .npy of FROZEN item embeddings "
                "(e.g. sentence_t5.npy — item text, no CF). When set, the recode is signal-GUIDED: "
                "items go to their nearest under-capacity code centroid (semantics-preserving) "
                "instead of the least-used code (entropy-only, semantics-blind). Empty = entropy-only."
            )
        },
    )
    num_beams: int = field(default=50, metadata={"help": "Beam width used by T5 generation."})
    eval_topk: tuple[int, ...] = field(
        default=(5, 10),
        metadata={"help": "Top-k values expected during TIGER full evaluation; used to validate num_beams."},
    )

    num_layers: int = field(default=4, metadata={"help": "Number of T5 encoder layers."})
    num_decoder_layers: int = field(default=4, metadata={"help": "Number of T5 decoder layers."})
    d_model: int = field(default=128, metadata={"help": "T5 hidden size."})
    d_ff: int = field(default=1024, metadata={"help": "T5 feed-forward hidden size."})
    num_heads: int = field(default=6, metadata={"help": "Number of T5 attention heads."})
    d_kv: int = field(default=64, metadata={"help": "T5 key/value projection size per head."})
    dropout_rate: float = field(default=0.1, metadata={"help": "T5 dropout rate."})
    activation_function: str = field(default="relu", metadata={"help": "T5 activation function."})
    feed_forward_proj: str = field(default="relu", metadata={"help": "T5 feed-forward projection type."})


__all__ = ["TIGERConfig"]
