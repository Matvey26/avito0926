"""Оценивает TF-IDF и BM25 на общей матрице частот термов.

Модуль обеспечивает чистоту сравнения: обе модели получают одну матрицу
CountVectorizer и одинаковые query-матрицы. Различаются только преобразование
весов документов и обработка частоты термов запроса.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

from retrieval import binarize_queries, build_bm25_matrix, evaluate_retrieval


def evaluate_models(
    counts: csr_matrix,
    query_counts: dict[str, csr_matrix],
    validations: dict[str, pd.DataFrame],
    relevance_rows: dict[str, dict[tuple[object, ...], frozenset[int]]],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Оценивает обе модели на одном словаре и наборе частот.

    Аргументы:
        counts: Матрица частот термов в документах.
        query_counts: Матрицы частот запросов по режимам валидации.
        validations: Валидационные таблицы по режимам.
        relevance_rows: Релевантные строки корпуса по режимам и сигнатурам.
        item_categories: Категории по строкам корпуса.
        item_ids: Item ID по строкам корпуса.

    Возвращает:
        Строки метрик TF-IDF и BM25 для всех режимов и диагностику времени и
        параметров преобразований.
    """
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}

    started = time.monotonic()
    transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    tfidf_documents = transformer.fit_transform(counts).astype(np.float32)
    tfidf_transposed = tfidf_documents.T.tocsr()
    del tfidf_documents
    for mode, validation in validations.items():
        tfidf_queries = transformer.transform(query_counts[mode]).astype(np.float32)
        rows.append(
            {
                "model": "tfidf",
                "mode": mode,
                **evaluate_retrieval(
                    tfidf_queries,
                    tfidf_transposed,
                    validation,
                    relevance_rows[mode],
                    item_categories,
                    item_ids,
                ),
            }
        )
        del tfidf_queries
    diagnostics["tfidf_seconds"] = time.monotonic() - started
    del transformer, tfidf_transposed
    gc.collect()

    started = time.monotonic()
    bm25_documents, bm25_details = build_bm25_matrix(counts)
    bm25_transposed = bm25_documents.T.tocsr()
    del bm25_documents
    for mode, validation in validations.items():
        bm25_queries = binarize_queries(query_counts[mode])
        rows.append(
            {
                "model": "bm25",
                "mode": mode,
                **evaluate_retrieval(
                    bm25_queries,
                    bm25_transposed,
                    validation,
                    relevance_rows[mode],
                    item_categories,
                    item_ids,
                ),
            }
        )
        del bm25_queries
    diagnostics["bm25_seconds"] = time.monotonic() - started
    diagnostics["bm25"] = bm25_details
    del bm25_transposed
    gc.collect()
    return rows, diagnostics
