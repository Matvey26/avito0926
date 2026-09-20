"""Строит изменённые word и char BM25-каналы эксперимента 0013.

Общая retrieval-реализация и протокол подготовки текстов переиспользуются из
неизменяемого эксперимента 0012; здесь зафиксирована только проверяемая дельта
диапазонов n-грамм.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from checkpointing import CandidateStore
from sparse_retrievers import build_sparse_channel
from text_processing import prepare_char_documents, prepare_word_documents

from settings_0013 import (
    CHAR_MAX_DF,
    CHAR_MAX_FEATURES,
    CHAR_NGRAM,
    MIN_DF,
    WORD_MAX_DF,
    WORD_MAX_FEATURES,
    WORD_NGRAM,
)


def ensure_word_variant(
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
    store: CandidateStore,
) -> None:
    """Строит word BM25 с лемматизированными униграммами и биграммами.

    Args:
        items: Корпус объявлений.
        validations: Cold/warm-запросы.
        item_categories: Категории строк корпуса.
        item_ids: Идентификаторы строк корпуса.
        store: Хранилище top-500 варианта.
    """
    name = "word_bm25_1_2"
    modes = tuple(validations)
    if store.complete(name, modes):
        print(f"{name}: checkpoint готов", flush=True)
        return
    print("Подготовка word BM25 (1,2)...", flush=True)
    documents, queries, cache_size = prepare_word_documents(items, validations)
    vectorizer = CountVectorizer(
        lowercase=False,
        ngram_range=WORD_NGRAM,
        min_df=MIN_DF,
        max_df=WORD_MAX_DF,
        max_features=WORD_MAX_FEATURES,
        dtype=np.float32,
    )
    build_sparse_channel(
        name,
        documents,
        queries,
        validations,
        item_categories,
        item_ids,
        vectorizer,
        {
            "normalization": "russian_lemma",
            "document_variant": "all_text",
            "word_ngram": list(WORD_NGRAM),
            "normalization_cache_size": cache_size,
        },
        store,
    )


def ensure_char_variant(
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
    store: CandidateStore,
) -> None:
    """Строит char BM25 с диапазоном символьных 3–4-грамм.

    Args:
        items: Корпус объявлений.
        validations: Cold/warm-запросы.
        item_categories: Категории строк корпуса.
        item_ids: Идентификаторы строк корпуса.
        store: Хранилище top-500 варианта.
    """
    name = "char_bm25_3_4"
    modes = tuple(validations)
    if store.complete(name, modes):
        print(f"{name}: checkpoint готов", flush=True)
        return
    print("Подготовка char BM25 (3,4)...", flush=True)
    documents, queries = prepare_char_documents(items, validations)
    vectorizer = CountVectorizer(
        analyzer="char_wb",
        lowercase=False,
        ngram_range=CHAR_NGRAM,
        min_df=MIN_DF,
        max_df=CHAR_MAX_DF,
        max_features=CHAR_MAX_FEATURES,
        dtype=np.float32,
    )
    build_sparse_channel(
        name,
        documents,
        queries,
        validations,
        item_categories,
        item_ids,
        vectorizer,
        {
            "normalization": "plain",
            "document_variant": "all_text",
            "char_ngram": list(CHAR_NGRAM),
        },
        store,
    )
