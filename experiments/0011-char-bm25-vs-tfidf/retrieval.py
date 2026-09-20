"""Строит BM25-веса и оценивает чистый char-retrieval.

TF-IDF и BM25 передают сюда sparse-матрицы одинаковой формы. Top-50
ограничивается целевой категорией, ничьи разрешаются по ``item_id``. Word-
признаки, локация, популярность и fallback намеренно отсутствуют.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from assets.common_code.metrics import count_relevant, summarize_recalls
from assets.common_code.validation import signature_key, target_category
from settings import BM25_B, BM25_K1, TOP_K


def build_bm25_matrix(counts: csr_matrix) -> tuple[csr_matrix, dict[str, float]]:
    """Преобразует частоты char-н-грамм в BM25-веса.

    Используется Okapi BM25 с положительным Robertson IDF:
    ``log(1 + (N - df + 0.5) / (df + 0.5))``.

    Args:
        counts: CSR-матрица ``документы × char-н-граммы``.

    Returns:
        BM25-взвешенную матрицу и диагностику длины документов и IDF.

    Raises:
        ValueError: Если матрица не содержит документов или термов.
    """
    if counts.shape[0] == 0 or counts.nnz == 0:
        raise ValueError("BM25 требует непустую матрицу частот")
    document_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average_length = float(document_lengths.mean())
    document_frequency = np.bincount(
        counts.indices, minlength=counts.shape[1]
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
    return weighted, {
        "average_document_length": average_length,
        "idf_min": float(idf.min()),
        "idf_max": float(idf.max()),
    }


def binarize_queries(counts: csr_matrix) -> csr_matrix:
    """Удаляет влияние повторов char-н-грамм в BM25-запросе.

    Args:
        counts: CSR-матрица частот в запросах.

    Returns:
        Копию матрицы со всеми ненулевыми значениями, равными единице.
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
    """Выбирает top-k нужной категории со стабильными ничьими.

    Args:
        scores: Однострочная sparse-матрица оценок корпуса.
        category: Целевая категория объявления.
        item_categories: Категории строк корпуса.
        item_ids: Идентификаторы строк корпуса.

    Returns:
        Номера строк по убыванию оценки и затем по item ID.
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
    """Считает Recall@50 одной модели на одном сплите.

    Args:
        query_matrix: Sparse-представления запросов.
        document_transposed: Транспонированная матрица документов.
        validation: Запросы в порядке строк ``query_matrix``.
        relevance_rows: Релевантные строки корпуса по сигнатурам.
        item_categories: Категории строк корпуса.
        item_ids: Item ID строк корпуса.

    Returns:
        Число запросов, средний Recall@50 и число запросов с попаданием.
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
