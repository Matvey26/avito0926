"""Shared text cleanup and word-branch morphological normalization."""

from __future__ import annotations

import re
import time
from typing import Callable

import pandas as pd
import pymorphy3
import snowballstemmer

from settings import NORMALIZATIONS


TOKEN_RE = re.compile(r"(?u)\b\w{2,}\b")
RUSSIAN_TOKEN_RE = re.compile(r"^[а-я]+$")


def normalize_text(text: object) -> str:
    """Apply only normalization shared by word and character branches."""
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


class TokenNormalizer:
    """Keep tokenization fixed while changing only the morphological transform."""

    def __init__(self, mode: str) -> None:
        if mode not in NORMALIZATIONS:
            raise ValueError(f"Unknown normalization: {mode}")
        self.mode = mode
        self.cache: dict[str, str] = {}
        self.stemmer = (
            snowballstemmer.stemmer("russian")
            if mode == "russian_snowball"
            else None
        )
        self.morph = pymorphy3.MorphAnalyzer() if mode == "russian_lemma" else None

    def _normalize_unseen(self, tokens: list[str]) -> None:
        if self.mode == "russian_snowball":
            assert self.stemmer is not None
            self.cache.update(zip(tokens, self.stemmer.stemWords(tokens)))
            return
        if self.mode == "russian_lemma":
            assert self.morph is not None
            for token in tokens:
                if RUSSIAN_TOKEN_RE.fullmatch(token):
                    lemma = self.morph.parse(token)[0].normal_form.replace("ё", "е")
                else:
                    lemma = token
                self.cache[token] = lemma
            return
        self.cache.update((token, token) for token in tokens)

    def __call__(self, text: object) -> str:
        tokens = TOKEN_RE.findall(normalize_text(text))
        unseen = list(dict.fromkeys(token for token in tokens if token not in self.cache))
        if unseen:
            self._normalize_unseen(unseen)
        return " ".join(self.cache[token] for token in tokens)


def preprocess_series(
    values: pd.Series, preprocessor: Callable[[object], str], label: str
) -> pd.Series:
    """Preprocess in visible chunks so a long lemmatization run reports progress."""
    started = time.monotonic()
    result: list[str] = []
    chunk_size = 25_000
    total = len(values)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        result.extend(preprocessor(value) for value in values.iloc[start:stop])
        print(f"  {label}: {stop:,}/{total:,}", flush=True)
    print(f"  {label} finished in {time.monotonic() - started:.1f}s", flush=True)
    return pd.Series(result, index=values.index, dtype="object")


def build_word_documents(
    items: pd.DataFrame, preprocessor: TokenNormalizer
) -> pd.Series:
    title = preprocess_series(items["item_title_raw"], preprocessor, "item titles")
    params = preprocess_series(items["item_infm_params_text"], preprocessor, "item params")
    description = preprocess_series(
        items["item_description_raw"], preprocessor, "item descriptions"
    )
    documents = title + " " + title + " " + title + " " + params + " " + description
    del title, params, description
    return documents


def build_word_queries(
    validation: pd.DataFrame, preprocessor: TokenNormalizer, label: str
) -> pd.Series:
    query = preprocess_series(
        validation["search_query"], preprocessor, f"{label} query text"
    )
    filters = preprocess_series(
        validation["search_infm_params_text"], preprocessor, f"{label} query filters"
    )
    result = query + " " + query + " " + filters
    del query, filters
    return result
