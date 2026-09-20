"""Строит два разреженных retrieval-канала итогового решения.

Word BM25 повторяет победителя эксперимента 0008, char BM25 — победителя
эксперимента 0011. Тяжёлые матрицы освобождаются между каналами.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from retrieval import CandidateBatch, build_bm25_matrix, retrieve_sparse
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


def build_channel(
    name: str,
    documents: pd.Series,
    query_documents: pd.Series,
    queries: pd.DataFrame,
    item_categories: np.ndarray,
    item_ids: np.ndarray,
    vectorizer: CountVectorizer,
) -> CandidateBatch:
    """Индексирует документы и извлекает BM25 top-500.

    Args:
        name: Название канала для сообщений прогресса.
        documents: Подготовленные тексты объявлений.
        query_documents: Подготовленные тексты запросов.
        queries: Исходные запросы с категориями.
        item_categories: Категории строк корпуса.
        item_ids: Идентификаторы строк корпуса.
        vectorizer: Настроенный CountVectorizer.

    Returns:
        Кандидатов и исходные BM25-score каждого запроса.
    """
    started = time.monotonic()
    counts = vectorizer.fit_transform(documents).tocsr()
    query_counts = vectorizer.transform(query_documents).tocsr()
    print(
        f"  {name}: словарь={len(vectorizer.vocabulary_):,}, "
        f"nnz={counts.nnz:,}, индекс={time.monotonic() - started:.1f} с",
        flush=True,
    )
    weighted = build_bm25_matrix(counts)
    result = retrieve_sparse(
        query_counts,
        weighted,
        queries,
        item_categories,
        item_ids,
    )
    print(f"  {name}: кандидаты готовы", flush=True)
    del counts, query_counts, weighted
    gc.collect()
    return result


def build_word_channel(
    items: pd.DataFrame,
    queries: pd.DataFrame,
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> CandidateBatch:
    """Строит лемматизированный unigram word BM25 по всем полям.

    Args:
        items: Benchmark-корпус объявлений.
        queries: Benchmark-запросы.
        item_categories: Категории строк корпуса.
        item_ids: Идентификаторы строк корпуса.

    Returns:
        Top-500 word BM25.
    """
    print("Подготовка word BM25...", flush=True)
    documents, query_documents = prepare_word_documents(items, queries)
    vectorizer = CountVectorizer(
        lowercase=False,
        ngram_range=(1, 1),
        min_df=WORD_MIN_DF,
        max_df=WORD_MAX_DF,
        max_features=WORD_MAX_FEATURES,
        dtype=np.float32,
    )
    result = build_channel(
        "word BM25",
        documents,
        query_documents,
        queries,
        item_categories,
        item_ids,
        vectorizer,
    )
    del documents, query_documents, vectorizer
    gc.collect()
    return result


def build_char_channel(
    items: pd.DataFrame,
    queries: pd.DataFrame,
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> CandidateBatch:
    """Строит char_wb BM25 (4, 4) по всем полям.

    Args:
        items: Benchmark-корпус объявлений.
        queries: Benchmark-запросы.
        item_categories: Категории строк корпуса.
        item_ids: Идентификаторы строк корпуса.

    Returns:
        Top-500 char BM25.
    """
    print("Подготовка char BM25...", flush=True)
    documents, query_documents = prepare_char_documents(items, queries)
    vectorizer = CountVectorizer(
        analyzer="char_wb",
        lowercase=False,
        ngram_range=CHAR_NGRAM,
        min_df=CHAR_MIN_DF,
        max_df=CHAR_MAX_DF,
        max_features=CHAR_MAX_FEATURES,
        dtype=np.float32,
    )
    result = build_channel(
        "char BM25",
        documents,
        query_documents,
        queries,
        item_categories,
        item_ids,
        vectorizer,
    )
    del documents, query_documents, vectorizer
    gc.collect()
    return result
