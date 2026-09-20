"""Строит word BM25 и char BM25 и сохраняет их top-500.

Word-конфигурация дословно повторяет победителя 0008, char-конфигурация —
победителя 0011. После извлечения кандидатов тяжёлые sparse-индексы удаляются.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from checkpointing import CandidateStore
from retrieval import binarize_queries, build_bm25_matrix, retrieve_sparse
from settings import (
    CHAR_MAX_DF,
    CHAR_MAX_FEATURES,
    CHAR_MIN_DF,
    CHAR_NGRAM,
    WORD_MAX_DF,
    WORD_MAX_FEATURES,
    WORD_MIN_DF,
)
from text_processing import prepare_char_documents, prepare_word_documents


def build_sparse_channel(
    name: str,
    documents: pd.Series,
    query_documents: dict[str, pd.Series],
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
    vectorizer: CountVectorizer,
    extra_diagnostics: dict[str, object],
    store: CandidateStore,
) -> None:
    """Индексирует один BM25-канал и сохраняет кандидатов.

    Args:
        name: Имя checkpoint-канала.
        documents: Документы объявлений.
        query_documents: Документы запросов по режимам.
        validations: Валидационные запросы.
        item_categories: Категории строк корпуса.
        item_ids: Item ID строк корпуса.
        vectorizer: Настроенный общий CountVectorizer.
        extra_diagnostics: Параметры подготовки текста.
        store: Хранилище top-500.

    Returns:
        Ничего.
    """
    started = time.monotonic()
    counts = vectorizer.fit_transform(documents).tocsr()
    query_counts = {
        mode: vectorizer.transform(values).tocsr()
        for mode, values in query_documents.items()
    }
    indexing_seconds = time.monotonic() - started
    print(
        f"  {name}: словарь={len(vectorizer.vocabulary_):,}, "
        f"nnz={counts.nnz:,}, индекс={indexing_seconds:.1f} с",
        flush=True,
    )
    started = time.monotonic()
    weighted, bm25_details = build_bm25_matrix(counts)
    transposed = weighted.T.tocsr()
    batches = {
        mode: retrieve_sparse(
            binarize_queries(query_counts[mode]),
            transposed,
            validations[mode],
            item_categories,
            item_ids,
        )
        for mode in validations
    }
    store.save(
        name,
        batches,
        {
            **extra_diagnostics,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "matrix_shape": list(counts.shape),
            "matrix_nnz": int(counts.nnz),
            "indexing_seconds": indexing_seconds,
            "retrieval_seconds": time.monotonic() - started,
            "bm25": bm25_details,
        },
    )
    del counts, query_counts, weighted, transposed, batches
    gc.collect()


def ensure_word_bm25(
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
    store: CandidateStore,
) -> None:
    """Строит отсутствующий word BM25 checkpoint из 0008.

    Args:
        items: Корпус объявлений.
        validations: Cold/warm-запросы.
        item_categories: Категории корпуса.
        item_ids: Item ID корпуса.
        store: Хранилище кандидатов.

    Returns:
        Ничего.
    """
    modes = tuple(validations)
    if store.complete("word_bm25", modes):
        print("word_bm25: checkpoint готов", flush=True)
        return
    print("Подготовка word BM25...", flush=True)
    documents, queries, cache_size = prepare_word_documents(items, validations)
    vectorizer = CountVectorizer(
        lowercase=False,
        ngram_range=(1, 1),
        min_df=WORD_MIN_DF,
        max_df=WORD_MAX_DF,
        max_features=WORD_MAX_FEATURES,
        dtype=np.float32,
    )
    build_sparse_channel(
        "word_bm25",
        documents,
        queries,
        validations,
        item_categories,
        item_ids,
        vectorizer,
        {"normalization": "russian_lemma", "document_variant": "all_text", "normalization_cache_size": cache_size},
        store,
    )


def ensure_char_bm25(
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
    store: CandidateStore,
) -> None:
    """Строит отсутствующий char BM25 checkpoint из 0011.

    Args:
        items: Корпус объявлений.
        validations: Cold/warm-запросы.
        item_categories: Категории корпуса.
        item_ids: Item ID корпуса.
        store: Хранилище кандидатов.

    Returns:
        Ничего.
    """
    modes = tuple(validations)
    if store.complete("char_bm25", modes):
        print("char_bm25: checkpoint готов", flush=True)
        return
    print("Подготовка char BM25...", flush=True)
    documents, queries = prepare_char_documents(items, validations)
    vectorizer = CountVectorizer(
        analyzer="char_wb",
        lowercase=False,
        ngram_range=CHAR_NGRAM,
        min_df=CHAR_MIN_DF,
        max_df=CHAR_MAX_DF,
        max_features=CHAR_MAX_FEATURES,
        dtype=np.float32,
    )
    build_sparse_channel(
        "char_bm25",
        documents,
        queries,
        validations,
        item_categories,
        item_ids,
        vectorizer,
        {"normalization": "plain", "document_variant": "all_text", "char_ngram": list(CHAR_NGRAM)},
        store,
    )
