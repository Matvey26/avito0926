"""Реализует BM25 и детерминированное извлечение top-500.

Sparse- и dense-каналы используют одинаковый фильтр категории и разрешают
ничьи по ``item_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from settings import (
    BM25_B,
    BM25_K1,
    CANDIDATE_POOL,
    FASTTEXT_QUERY_BATCH_SIZE,
    target_category,
)


@dataclass(frozen=True)
class CandidateBatch:
    """Хранит top-500 и исходные score всех запросов."""

    indices: np.ndarray
    scores: np.ndarray
    lengths: np.ndarray

    def row(self, number: int) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает кандидатов одного запроса без padding.

        Args:
            number: Номер запроса.

        Returns:
            Номера строк корпуса и score.
        """
        length = int(self.lengths[number])
        return self.indices[number, :length], self.scores[number, :length]


def empty_batch(query_count: int) -> CandidateBatch:
    """Создаёт пустое прямоугольное хранилище кандидатов.

    Args:
        query_count: Число запросов.

    Returns:
        Batch с padding ``-1``.
    """
    return CandidateBatch(
        np.full((query_count, CANDIDATE_POOL), -1, dtype=np.int32),
        np.zeros((query_count, CANDIDATE_POOL), dtype=np.float32),
        np.zeros(query_count, dtype=np.int16),
    )


def build_bm25_matrix(counts: csr_matrix) -> csr_matrix:
    """Преобразует частоты документов в Okapi BM25-веса.

    Args:
        counts: CSR-матрица ``документы × признаки``.

    Returns:
        BM25-взвешенную CSR-матрицу.
    """
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average_length = float(lengths.mean())
    document_frequency = np.bincount(
        counts.indices, minlength=counts.shape[1]
    ).astype(np.float32)
    idf = np.log1p(
        (counts.shape[0] - document_frequency + 0.5)
        / (document_frequency + 0.5)
    ).astype(np.float32)
    weighted = counts.astype(np.float32, copy=True)
    row_norm = np.repeat(
        BM25_K1 * (1.0 - BM25_B + BM25_B * lengths / average_length),
        np.diff(weighted.indptr),
    )
    weighted.data *= BM25_K1 + 1.0
    weighted.data /= weighted.data / (BM25_K1 + 1.0) + row_norm
    weighted.data *= idf[weighted.indices]
    weighted.eliminate_zeros()
    return weighted


def stable_top(
    indices: np.ndarray, scores: np.ndarray, item_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Выбирает top-500 по score с детерминированными ничьими.

    Args:
        indices: Номера строк рассматриваемого корпуса.
        scores: Их оценки.
        item_ids: Item ID в системе координат ``indices``.

    Returns:
        Индексы и score по убыванию, затем по item ID.
    """
    if len(indices) > CANDIDATE_POOL:
        threshold = np.partition(scores, -CANDIDATE_POOL)[-CANDIDATE_POOL]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        remaining = CANDIDATE_POOL - len(above)
        tied_order = np.argsort(item_ids[indices[tied]], kind="stable")[:remaining]
        selected = np.concatenate([above, tied[tied_order]])
        indices, scores = indices[selected], scores[selected]
    order = np.lexsort((item_ids[indices], -scores))
    return indices[order].astype(np.int32), scores[order].astype(np.float32)


def retrieve_sparse(
    query_counts: csr_matrix,
    documents: csr_matrix,
    queries: pd.DataFrame,
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> CandidateBatch:
    """Извлекает BM25 top-500 внутри категории.

    Args:
        query_counts: Частоты признаков в запросах.
        documents: BM25-взвешенные документы.
        queries: Benchmark-запросы.
        item_categories: Категории корпуса.
        item_ids: Item ID корпуса.

    Returns:
        Кандидатов и исходные BM25-score.
    """
    binary_queries = query_counts.astype(np.float32, copy=True)
    binary_queries.data.fill(1.0)
    transposed = documents.T.tocsr()
    batch = empty_batch(len(queries))
    for number, (_, query) in enumerate(queries.iterrows()):
        row = (binary_queries[number] @ transposed).tocsr()
        keep = item_categories[row.indices] == target_category(query["search_category"])
        indices, scores = stable_top(row.indices[keep], row.data[keep], item_ids)
        length = len(indices)
        batch.indices[number, :length] = indices
        batch.scores[number, :length] = scores
        batch.lengths[number] = length
    return batch


def retrieve_dense(
    item_vectors: np.ndarray,
    query_vectors: np.ndarray,
    queries: pd.DataFrame,
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> CandidateBatch:
    """Извлекает точный cosine top-500 внутри категории.

    Args:
        item_vectors: L2-нормализованные объявления.
        query_vectors: L2-нормализованные запросы.
        queries: Benchmark-запросы.
        item_categories: Категории корпуса.
        item_ids: Item ID корпуса.

    Returns:
        Dense-кандидатов и cosine score.
    """
    batch = empty_batch(len(queries))
    categories = np.asarray(
        [target_category(value) for value in queries["search_category"]],
        dtype=np.int32,
    )
    for category in sorted(set(categories)):
        item_rows = np.flatnonzero(item_categories == category)
        query_rows = np.flatnonzero(categories == category)
        category_vectors = item_vectors[item_rows]
        category_ids = item_ids[item_rows]
        for start in range(0, len(query_rows), FASTTEXT_QUERY_BATCH_SIZE):
            rows = query_rows[start : start + FASTTEXT_QUERY_BATCH_SIZE]
            matrix = query_vectors[rows] @ category_vectors.T
            for local_number, query_row in enumerate(rows):
                local, scores = stable_top(
                    np.arange(len(item_rows)), matrix[local_number], category_ids
                )
                indices = item_rows[local]
                length = len(indices)
                batch.indices[query_row, :length] = indices
                batch.scores[query_row, :length] = scores
                batch.lengths[query_row] = length
    return batch
