"""Извлекает top-500 из sparse- и dense-матриц.

Оба пути применяют одинаковый фильтр категории и стабильное разрешение ничьих
по ``item_id``. Возвращаются не только строки корпуса, но и исходные score,
необходимые последующему fusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from assets.common_code.validation import target_category
from checkpointing import CandidateBatch, empty_batch
from settings import BM25_B, BM25_K1, CANDIDATE_POOL, FASTTEXT_QUERY_BATCH_SIZE


def build_bm25_matrix(counts: csr_matrix) -> tuple[csr_matrix, dict[str, float]]:
    """Преобразует матрицу частот в Okapi BM25-веса.

    Args:
        counts: CSR-матрица частот ``документы × признаки``.

    Returns:
        BM25-взвешенную матрицу и диагностику длины документов и IDF.

    Raises:
        ValueError: Если матрица пуста.
    """
    if counts.shape[0] == 0 or counts.nnz == 0:
        raise ValueError("BM25 требует непустую матрицу частот")
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
    length_norm = BM25_K1 * (
        1.0 - BM25_B + BM25_B * lengths / average_length
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
    """Преобразует ненулевые частоты запроса в единицы.

    Args:
        counts: CSR-матрица запросов.

    Returns:
        Копию с бинарными значениями для классического BM25.
    """
    result = counts.astype(np.float32, copy=True)
    result.data.fill(1.0)
    return result


def stable_top(
    indices: np.ndarray, scores: np.ndarray, item_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Выбирает top-500 со стабильным разрешением ничьих.

    Args:
        indices: Номера строк корпуса.
        scores: Оценки тех же строк.
        item_ids: Все item ID корпуса.

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
        indices = indices[selected]
        scores = scores[selected]
    order = np.lexsort((item_ids[indices], -scores))
    return (
        indices[order].astype(np.int32, copy=False),
        scores[order].astype(np.float32, copy=False),
    )


def retrieve_sparse(
    query_matrix: csr_matrix,
    document_transposed: csr_matrix,
    validation: pd.DataFrame,
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> CandidateBatch:
    """Извлекает sparse top-500 внутри целевой категории.

    Args:
        query_matrix: BM25-запросы.
        document_transposed: Транспонированные BM25-документы.
        validation: Запросы в порядке строк матрицы.
        item_categories: Категории строк корпуса.
        item_ids: Item ID строк корпуса.

    Returns:
        Прямоугольный batch кандидатов и исходных BM25-score.
    """
    batch = empty_batch(len(validation))
    for row_number, (_, query) in enumerate(validation.iterrows()):
        score_row = (query_matrix[row_number] @ document_transposed).tocsr()
        keep = item_categories[score_row.indices] == target_category(
            query["search_category"]
        )
        indices, scores = stable_top(
            score_row.indices[keep], score_row.data[keep], item_ids
        )
        length = len(indices)
        batch.indices[row_number, :length] = indices
        batch.scores[row_number, :length] = scores
        batch.lengths[row_number] = length
    return batch


def retrieve_dense(
    item_embeddings: np.ndarray,
    query_embeddings: dict[str, np.ndarray],
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> dict[str, CandidateBatch]:
    """Извлекает точный cosine top-500 внутри категории.

    Args:
        item_embeddings: L2-нормализованные векторы объявлений.
        query_embeddings: L2-нормализованные векторы запросов по сплитам.
        validations: Валидационные запросы по сплитам.
        item_categories: Категории строк корпуса.
        item_ids: Item ID строк корпуса.

    Returns:
        Dense-кандидаты по режимам.
    """
    batches = {mode: empty_batch(len(frame)) for mode, frame in validations.items()}
    query_categories = {
        mode: np.asarray(
            [target_category(value) for value in frame["search_category"]],
            dtype=np.int32,
        )
        for mode, frame in validations.items()
    }
    categories = sorted(set(np.concatenate(list(query_categories.values()))))
    for category in categories:
        item_rows = np.flatnonzero(item_categories == category)
        if not len(item_rows):
            continue
        category_vectors = item_embeddings[item_rows]
        category_ids = item_ids[item_rows]
        for mode in validations:
            query_rows = np.flatnonzero(query_categories[mode] == category)
            for start in range(0, len(query_rows), FASTTEXT_QUERY_BATCH_SIZE):
                rows = query_rows[start : start + FASTTEXT_QUERY_BATCH_SIZE]
                score_matrix = query_embeddings[mode][rows] @ category_vectors.T
                for batch_number, query_row in enumerate(rows):
                    local, scores = stable_top(
                        np.arange(len(item_rows)), score_matrix[batch_number], category_ids
                    )
                    indices = item_rows[local]
                    length = len(indices)
                    batches[mode].indices[query_row, :length] = indices
                    batches[mode].scores[query_row, :length] = scores
                    batches[mode].lengths[query_row] = length
        del category_vectors
    return batches
