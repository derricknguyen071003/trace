"""
c-TF-IDF text cleaning for lexical SID phrase labeling.

Text cleaning pipeline that turns each item's raw metadata sentence into a
space-normalised, de-duplicated *set* of content words used as the labeling
corpus for the codebooks (Stage C, see README).

This module is shared by lexical C1PP phrase-artifact construction.
"""

import re
from collections import Counter

# NLTK is part of the project env (see env.yml). Imported lazily / guarded so a
# missing corpus degrades to a still-usable (if slightly coarser) cleaner rather
# than crashing a training run.
try:
    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.stem import WordNetLemmatizer as _WordNetLemmatizer

    _STOPWORDS = set(_nltk_stopwords.words("english"))
    _LEMMATIZER = _WordNetLemmatizer()
except Exception:  # pragma: no cover - fallback if corpora are unavailable
    _STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
        "is", "are", "this", "that", "it", "as", "by", "at", "from", "be",
    }
    _LEMMATIZER = None

# Words carried by (almost) every item -- pure noise for labeling.
TEMPLATE_WORDS = {"title", "brand", "category", "categories", "price"}
ARTIFACTS = {"unknown", "amp", "tm", "nan"}
UNITS = {"oz", "ounce", "fl", "ml", "mm", "pc", "fluid"}

# Words derived from a split's root category -- present in ~every item of that
# split, so they carry no signal. Keyed by lowercased dataset name.
DOMAIN_WORDS = {
    "beauty": {"beauty"},
    "grocery_and_gourmet_food": {"grocery", "gourmet", "food"},
    "grocery": {"grocery", "gourmet", "food"},
    "pet_supplies": {"pet", "supply", "supplies"},
    "video_games": {"video", "game", "games"},
    "toys_and_games": {"toy", "toys", "game", "games"},
    "toys": {"toy", "toys", "game", "games"},
    "sports_and_outdoors": {"sport", "sports", "outdoor", "outdoors"},
    "sports": {"sport", "sports", "outdoor", "outdoors"},
}

_TOKEN_RE = re.compile(r"[a-z]+")


def _lemmatize(word):
    if _LEMMATIZER is None:
        return word
    return _LEMMATIZER.lemmatize(word)


def clean_texts(texts, dataset, min_doc_freq=5):
    """Turn raw metadata sentences into per-item ordered word *sets*.

    Pipeline (matches the porting spec):
      1. lowercase, keep alphabetic tokens only
      2. drop NLTK English stopwords
      3. WordNet lemmatization
      4. drop template / artifact / unit / 1-char tokens
      5. drop dataset domain words
      6. drop words with corpus document-frequency < ``min_doc_freq``
      7. per-item dedup (keep first-occurrence order)

    Args:
        texts: list of raw metadata strings, row ``i`` == item ``i``.
        dataset: dataset name (e.g. ``"Beauty"``) for domain-word removal.
        min_doc_freq: minimum corpus document frequency to keep a word.

    Returns:
        list (len == len(texts)) of lists of words (deduped, order-preserving).
    """
    domain = DOMAIN_WORDS.get(dataset.lower(), set())
    drop = _STOPWORDS | TEMPLATE_WORDS | ARTIFACTS | UNITS | domain

    per_item = []
    doc_freq = Counter()
    for text in texts:
        seen = []
        seen_set = set()
        for raw in _TOKEN_RE.findall((text or "").lower()):
            if raw in drop:
                continue
            word = _lemmatize(raw)
            if len(word) <= 1 or word in drop:
                continue
            if word in seen_set:
                continue
            seen_set.add(word)
            seen.append(word)
        per_item.append(seen)
        for word in seen_set:
            doc_freq[word] += 1

    # step 6: corpus document-frequency filter (frequency == doc-count because
    # each item's words are deduped -- load-bearing for the c-TF-IDF math).
    cleaned = [[w for w in words if doc_freq[w] >= min_doc_freq] for words in per_item]
    return cleaned
