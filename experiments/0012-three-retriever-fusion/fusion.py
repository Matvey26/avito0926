"""Объединяет три top-500 пула двумя стратегиями fusion.

RRF использует только ранги, mean-minmax — среднее per-retriever score после
min-max нормализации. Затем fusion-score приводится к ``[0,1]`` и к нему один
раз добавляется фиксированный location bonus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from assets.common_code.metrics import count_relevant, summarize_recalls
from assets.common_code.validation import signature_key
from checkpointing import CandidateBatch
from settings import (
    FUSION_STRATEGIES,
    LOCATION_BONUS,
    RETRIEVERS,
    RRF_K,
    TOP_K,
)


def minmax(values: np.ndarray) -> np.ndarray:
    """Нормализует одномерные score в диапазон ``[0,1]``.

    Args:
        values: Исходные score одного retrieval-пула или fusion.

    Returns:
        Нормализованные ``float32``; константный массив превращается в нули.
    """
    if not len(values):
        return values.astype(np.float32, copy=True)
    minimum = float(values.min())
    span = float(values.max()) - minimum
    if span <= np.finfo(np.float32).eps:
        return np.zeros(len(values), dtype=np.float32)
    return ((values - minimum) / span).astype(np.float32, copy=False)


def union_with_inverse(
    rows: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Строит объединение кандидатов и отображения локальных строк.

    Args:
        rows: Кандидаты и score всех retrieval-каналов.

    Returns:
        Отсортированные уникальные строки корпуса и позиции каждого пула в
        этом объединении.
    """
    nonempty = [indices for indices, _ in rows if len(indices)]
    if not nonempty:
        return np.empty(0, dtype=np.int32), [np.empty(0, dtype=np.int32) for _ in rows]
    concatenated = np.concatenate(nonempty)
    union = np.unique(concatenated).astype(np.int32, copy=False)
    inverse = [
        np.searchsorted(union, indices).astype(np.int32, copy=False)
        for indices, _ in rows
    ]
    return union, inverse


def reciprocal_rank_fusion(
    rows: list[tuple[np.ndarray, np.ndarray]], inverse: list[np.ndarray], size: int
) -> np.ndarray:
    """Вычисляет RRF-score по рангам трёх retrieval-каналов.

    Args:
        rows: Отсортированные top-500 каждого канала.
        inverse: Их позиции в объединении кандидатов.
        size: Размер объединения.

    Returns:
        RRF-score, min-max нормализованный по текущему запросу.
    """
    fused = np.zeros(size, dtype=np.float32)
    for (indices, _), positions in zip(rows, inverse):
        ranks = np.arange(1, len(indices) + 1, dtype=np.float32)
        fused[positions] += 1.0 / (RRF_K + ranks)
    return minmax(fused)


def mean_minmax_fusion(
    rows: list[tuple[np.ndarray, np.ndarray]], inverse: list[np.ndarray], size: int
) -> np.ndarray:
    """Усредняет per-retriever min-max score, считая пропуски нулями.

    Args:
        rows: Top-500 и исходные score каждого канала.
        inverse: Позиции строк каналов в объединении.
        size: Размер объединения.

    Returns:
        Средний score в диапазоне ``[0,1]`` с фиксированным делителем три.
    """
    fused = np.zeros(size, dtype=np.float32)
    for (_, scores), positions in zip(rows, inverse):
        fused[positions] += minmax(scores)
    return fused / len(RETRIEVERS)


def select_fused(
    rows: list[tuple[np.ndarray, np.ndarray]],
    strategy: str,
    query_location: int,
    item_locations: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    """Применяет fusion, location bonus и выбирает итоговый top-50.

    Args:
        rows: Кандидаты трёх каналов одного запроса.
        strategy: ``rrf`` или ``mean_minmax``.
        query_location: Локация запроса.
        item_locations: Локации строк корпуса.
        item_ids: Item ID строк корпуса для стабильных ничьих.

    Returns:
        До 50 строк корпуса в порядке итогового score.

    Raises:
        ValueError: Если стратегия неизвестна.
    """
    union, inverse = union_with_inverse(rows)
    if not len(union):
        return union
    if strategy == "rrf":
        scores = reciprocal_rank_fusion(rows, inverse, len(union))
    elif strategy == "mean_minmax":
        scores = mean_minmax_fusion(rows, inverse, len(union))
    else:
        raise ValueError(f"Неизвестная стратегия fusion: {strategy}")
    scores += (item_locations[union] == query_location).astype(np.float32) * LOCATION_BONUS
    if len(union) > TOP_K:
        threshold = np.partition(scores, -TOP_K)[-TOP_K]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        remaining = TOP_K - len(above)
        tied_order = np.argsort(item_ids[union[tied]], kind="stable")[:remaining]
        selected = np.concatenate([above, tied[tied_order]])
        union = union[selected]
        scores = scores[selected]
    order = np.lexsort((item_ids[union], -scores))
    return union[order]


def evaluate_fusion(
    batches: dict[str, dict[str, CandidateBatch]],
    validations: dict[str, pd.DataFrame],
    relevance_rows: dict[str, dict[tuple[object, ...], frozenset[int]]],
    item_locations: np.ndarray,
    item_ids: np.ndarray,
) -> list[dict[str, object]]:
    """Оценивает обе fusion-стратегии на cold/warm-сплитах.

    Args:
        batches: ``retriever -> mode -> CandidateBatch``.
        validations: Валидационные запросы.
        relevance_rows: Релевантные строки корпуса по сигнатурам.
        item_locations: Локации корпуса.
        item_ids: Item ID корпуса.

    Returns:
        По одной строке Recall@50 на стратегию и режим.
    """
    rows: list[dict[str, object]] = []
    for strategy in FUSION_STRATEGIES:
        for mode, validation in validations.items():
            recalls: list[float] = []
            for row_number, (_, query) in enumerate(validation.iterrows()):
                candidate_rows = [
                    batches[name][mode].row(row_number) for name in RETRIEVERS
                ]
                predicted = select_fused(
                    candidate_rows,
                    strategy,
                    int(query["search_location_id"]),
                    item_locations,
                    item_ids,
                )
                relevant = relevance_rows[mode][signature_key(query)]
                recalls.append(count_relevant(predicted, relevant) / len(relevant))
            rows.append({"strategy": strategy, "mode": mode, **summarize_recalls(recalls)})
    return rows


def evaluate_retrievers(
    batches: dict[str, dict[str, CandidateBatch]],
    validations: dict[str, pd.DataFrame],
    relevance_rows: dict[str, dict[tuple[object, ...], frozenset[int]]],
) -> list[dict[str, object]]:
    """Считает Recall@50 и полноту полных candidate pool.

    Args:
        batches: Кандидаты по retrieval-каналам и режимам.
        validations: Валидационные запросы.
        relevance_rows: Релевантные строки корпуса по сигнатурам.

    Returns:
        Диагностические метрики каналов и объединённого пула. Для
        неранжированного объединения Recall@50 не определяется и равен NaN.
    """
    result: list[dict[str, object]] = []
    names = (*RETRIEVERS, "union_pool")
    for mode, validation in validations.items():
        top50 = {name: [] for name in RETRIEVERS}
        full_pool = {name: [] for name in names}
        for row_number, (_, query) in enumerate(validation.iterrows()):
            relevant = relevance_rows[mode][signature_key(query)]
            channel_indices = [
                batches[name][mode].row(row_number)[0] for name in RETRIEVERS
            ]
            for name, indices in zip(RETRIEVERS, channel_indices):
                top50[name].append(count_relevant(indices[:50], relevant) / len(relevant))
                full_pool[name].append(count_relevant(indices, relevant) / len(relevant))
            union = np.unique(np.concatenate(channel_indices))
            value = count_relevant(union, relevant) / len(relevant)
            full_pool["union_pool"].append(value)
        for name in names:
            result.append(
                {
                    "retriever": name,
                    "mode": mode,
                    "queries": len(validation),
                    "recall_at_50": (
                        float(np.mean(top50[name])) if name in RETRIEVERS else np.nan
                    ),
                    "recall_at_pool": float(np.mean(full_pool[name])),
                }
            )
    return result
