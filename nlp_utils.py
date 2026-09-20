"""NLTK-based tokenization and query normalization."""

from __future__ import annotations

import re
from functools import lru_cache

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
except ImportError:  # pragma: no cover
    nltk = None
    stopwords = None
    PorterStemmer = None
    word_tokenize = None


STEMMER = PorterStemmer() if PorterStemmer else None

SYNONYMS = {
    "btech": "btech",
    "b.tech": "btech",
    "be": "btech",
    "bca": "bca",
    "mca": "mca",
    "cse": "cse",
    "ece": "ece",
    "mech": "me",
    "mechanical": "me",
    "fee": "fee",
    "fees": "fee",
    "tuition": "fee",
    "hostel": "hostel",
    "exam": "exam",
    "exams": "exam",
    "admission": "admission",
    "apply": "admission",
    "library": "library",
    "hod": "faculty",
    "professor": "faculty",
    "faculty": "faculty",
    "placement": "placement",
    "scholarship": "scholarship",
    "fest": "events",
    "event": "events",
    "contact": "contact",
    "phone": "contact",
    "email": "contact",
}


def ensure_nltk_data() -> None:
    if nltk is None:
        return
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _stop_words() -> set[str]:
    ensure_nltk_data()
    if stopwords is None:
        return set()
    return set(stopwords.words("english"))


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s.%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    try:
        ensure_nltk_data()
        tokens = word_tokenize(normalized) if word_tokenize else normalized.split()
        stop = _stop_words()
    except Exception:
        tokens = normalized.split()
        stop = set()
    result: list[str] = []
    for token in tokens:
        if token in stop or len(token) < 2:
            continue
        stem = STEMMER.stem(token) if STEMMER else token
        mapped = SYNONYMS.get(stem, stem)
        result.append(mapped)
    return result


def keyword_overlap(query_tokens: list[str], target_tokens: list[str]) -> float:
    if not query_tokens or not target_tokens:
        return 0.0
    q = set(query_tokens)
    t = set(target_tokens)
    return len(q & t) / max(len(q), 1)


def contains_any(tokens: list[str], keywords: list[str]) -> bool:
    token_set = set(tokens)
    return any(k in token_set for k in keywords)
