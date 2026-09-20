"""Оценивает char TF-IDF и BM25 на общей матрице частот.

Обе модели используют один CountVectorizer и одинаковые query-матрицы.
Различаются только взвешивание документов и обработка частоты термов запроса.
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
    """Оценивает TF-IDF и BM25 на общем char-индексе.

    Args:
        counts: Частоты char-н-грамм в документах.
        query_counts: Частоты char-н-грамм в запросах по сплитам.
        validations: Валидационные запросы по сплитам.
        relevance_rows: Релевантные строки корпуса по сигнатурам.
        item_categories: Категории строк корпуса.
        item_ids: Item ID строк корпуса.

    Returns:
        Метрики двух моделей по сплитам и диагностику времени и BM25.
    """
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}

    started = time.monotonic()
    transformer = TfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True
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
