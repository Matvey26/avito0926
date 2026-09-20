"""Выполняет точный cosine dense retrieval внутри категории.

Эмбеддинги заранее L2-нормализованы, поэтому матричное произведение равно
cosine similarity. Батчи запросов не допускают построения полной матрицы
``запросы × объявления`` и сохраняют приемлемое потребление памяти.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from assets.common_code.metrics import count_relevant, summarize_recalls
from assets.common_code.validation import signature_key, target_category
from settings import QUERY_BATCH_SIZE, TOP_K


def stable_top_k(scores: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    """Выбирает top-k оценок со стабильным разрешением ничьих.

    Args:
        scores: Cosine-оценки объявлений одной категории.
        item_ids: Item ID в том же порядке.

    Returns:
        Локальные индексы кандидатов по убыванию оценки, затем по item ID.
    """
    if len(scores) <= TOP_K:
        selected = np.arange(len(scores))
    else:
        threshold = np.partition(scores, -TOP_K)[-TOP_K]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        remaining = TOP_K - len(above)
        tied_order = np.argsort(item_ids[tied], kind="stable")[:remaining]
        selected = np.concatenate([above, tied[tied_order]])
    order = np.lexsort((item_ids[selected], -scores[selected]))
    return selected[order]


def evaluate_dense_splits(
    item_embeddings: np.ndarray,
    query_embeddings: dict[str, np.ndarray],
    validations: dict[str, pd.DataFrame],
    relevance_rows: dict[str, dict[tuple[object, ...], frozenset[int]]],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
) -> list[dict[str, object]]:
    """Оценивает dense retrieval на cold- и warm-сплитах.

    Args:
        item_embeddings: Единичные векторы объявлений.
        query_embeddings: Единичные векторы запросов по режимам.
        validations: Валидационные запросы в порядке векторов.
        relevance_rows: Релевантные строки корпуса по сигнатурам.
        item_categories: Категории строк корпуса.
        item_ids: Item ID строк корпуса.

    Returns:
        По одной сводной строке Recall@50 на режим.
    """
    recalls = {
        mode: np.zeros(len(validation), dtype=np.float64)
        for mode, validation in validations.items()
    }
    query_categories = {
        mode: np.asarray(
            [target_category(value) for value in validation["search_category"]],
            dtype=np.int32,
        )
        for mode, validation in validations.items()
    }
    categories = sorted(set(np.concatenate(list(query_categories.values()))))

    for category in categories:
        item_rows = np.flatnonzero(item_categories == category)
        if not len(item_rows):
            continue
        category_embeddings = np.asarray(item_embeddings[item_rows])
        category_ids = item_ids[item_rows]
        for mode, validation in validations.items():
            query_rows = np.flatnonzero(query_categories[mode] == category)
            for start in range(0, len(query_rows), QUERY_BATCH_SIZE):
                batch_rows = query_rows[start : start + QUERY_BATCH_SIZE]
                scores = query_embeddings[mode][batch_rows] @ category_embeddings.T
                for batch_number, query_row in enumerate(batch_rows):
                    local = stable_top_k(scores[batch_number], category_ids)
                    predicted = item_rows[local]
                    query = validation.iloc[query_row]
                    relevant = relevance_rows[mode][signature_key(query)]
                    recalls[mode][query_row] = count_relevant(predicted, relevant) / len(relevant)
        del category_embeddings

    return [
        {"mode": mode, **summarize_recalls(values)}
        for mode, values in recalls.items()
    ]
