"""Строит BM25-веса и оценивает word-retrieval без дополнительных сигналов.

BM25 и TF-IDF передают в этот модуль sparse-матрицы одинаковой формы. Выбор
top-50 ограничивается целевой категорией, а ничьи разрешаются по ``item_id``.
Локация, char-признаки, популярность и fallback здесь намеренно не участвуют.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from assets.common_code.metrics import count_relevant, summarize_recalls
from assets.common_code.validation import signature_key, target_category
from settings import BM25_B, BM25_K1, TOP_K


def build_bm25_matrix(counts: csr_matrix) -> tuple[csr_matrix, dict[str, float]]:
    """Преобразует матрицу частот документов в матрицу BM25-весов.

    Используется классическая формула Okapi BM25 с положительным Robertson IDF:
    ``log(1 + (N - df + 0.5) / (df + 0.5))``. Коэффициенты ``k1`` и ``b``
    фиксированы в настройках эксперимента.

    Аргументы:
        counts: CSR-матрица частот термов размера «документы × признаки».

    Возвращает:
        BM25-взвешенную CSR-матрицу и диагностику средней длины документа и
        диапазона IDF.

    Исключения:
        ValueError: Если матрица не содержит документов или ненулевых термов.
    """
    if counts.shape[0] == 0 or counts.nnz == 0:
        raise ValueError("BM25 требует непустую матрицу частот")

    document_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average_length = float(document_lengths.mean())
    document_frequency = np.bincount(
        counts.indices,
        minlength=counts.shape[1],
    ).astype(np.float32)
    idf = np.log1p(
        (counts.shape[0] - document_frequency + 0.5)
        / (document_frequency + 0.5)
    ).astype(np.float32)

    weighted = counts.astype(np.float32, copy=True)
    length_norm = BM25_K1 * (
        1.0 - BM25_B + BM25_B * document_lengths / average_length
    )
    row_norm = np.repeat(length_norm, np.diff(weighted.indptr))
    weighted.data *= BM25_K1 + 1.0
    weighted.data /= weighted.data / (BM25_K1 + 1.0) + row_norm
    weighted.data *= idf[weighted.indices]
    weighted.eliminate_zeros()

    diagnostics = {
        "average_document_length": average_length,
        "idf_min": float(idf.min()),
        "idf_max": float(idf.max()),
    }
    return weighted, diagnostics


def binarize_queries(counts: csr_matrix) -> csr_matrix:
    """Удаляет влияние повторов терма в запросе для классического BM25.

    Аргументы:
        counts: CSR-матрица частот термов в запросах.

    Возвращает:
        Копию матрицы, в которой все ненулевые значения равны единице.
    """
    result = counts.astype(np.float32, copy=True)
    result.data.fill(1.0)
    return result


def stable_top_k(
    scores: csr_matrix,
    category: int,
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    """Выбирает top-k нужной категории со стабильным разрешением ничьих.

    Аргументы:
        scores: Однострочная sparse-матрица оценок по корпусу.
        category: Целевая категория объявления.
        item_categories: Категории по строкам корпуса.
        item_ids: Идентификаторы по строкам корпуса.

    Возвращает:
        Номера не более чем ``TOP_K`` строк корпуса, упорядоченные по убыванию
        оценки и затем по возрастанию ``item_id``.
    """
    indices = scores.indices
    values = scores.data
    if len(indices):
        keep = item_categories[indices] == category
        indices = indices[keep]
        values = values[keep]
    if not len(indices):
        return np.empty(0, dtype=np.int32)

    if len(indices) > TOP_K:
        threshold = np.partition(values, -TOP_K)[-TOP_K]
        above = np.flatnonzero(values > threshold)
        tied = np.flatnonzero(values == threshold)
        remaining = TOP_K - len(above)
        tied_order = np.argsort(item_ids[indices[tied]], kind="stable")[:remaining]
        selected = np.concatenate([above, tied[tied_order]])
        indices = indices[selected]
        values = values[selected]

    order = np.lexsort((item_ids[indices], -values))
    return indices[order].astype(np.int32, copy=False)


def evaluate_retrieval(
    query_matrix: csr_matrix,
    document_transposed: csr_matrix,
    validation: pd.DataFrame,
    relevance_rows: dict[tuple[object, ...], frozenset[int]],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> dict[str, object]:
    """Оценивает одну матрицу retrieval на одном валидационном режиме.

    Аргументы:
        query_matrix: Sparse-представления запросов.
        document_transposed: Транспонированная матрица документов той же модели.
        validation: Запросы в том же порядке, что строки ``query_matrix``.
        relevance_rows: Релевантные номера строк по сигнатурам запросов.
        item_categories: Категории по строкам корпуса.
        item_ids: Item ID по строкам корпуса для разрешения ничьих.

    Возвращает:
        Сводку с числом запросов, Recall@50 и числом запросов с попаданием.
    """
    recalls: list[float] = []
    for row_number, (_, query) in enumerate(validation.iterrows()):
        scores = (query_matrix[row_number] @ document_transposed).tocsr()
        predicted = stable_top_k(
            scores,
            target_category(query["search_category"]),
            item_categories,
            item_ids,
        )
        relevant = relevance_rows[signature_key(query)]
        recalls.append(count_relevant(predicted, relevant) / len(relevant))
    return summarize_recalls(recalls)
