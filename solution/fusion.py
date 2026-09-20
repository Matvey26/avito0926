"""Объединяет три retrieval-канала стратегии mean-minmax.

Для каждого запроса исходные score каждого канала отдельно приводятся к
``[0, 1]``. Пропущенные кандидаты получают ноль, три значения усредняются,
после чего к совпадающей локации прибавляется 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retrieval import CandidateBatch
from settings import LOCATION_BONUS, TOP_K


def minmax(values: np.ndarray) -> np.ndarray:
    """Нормализует score одного канала в диапазон ``[0, 1]``.

    Args:
        values: Оценки кандидатов одного retrieval-канала.

    Returns:
        Нормализованные float32; константный массив превращается в нули.
    """
    if not len(values):
        return values.astype(np.float32, copy=True)
    minimum = float(values.min())
    span = float(values.max()) - minimum
    if span <= np.finfo(np.float32).eps:
        return np.zeros(len(values), dtype=np.float32)
    return ((values - minimum) / span).astype(np.float32, copy=False)


def fuse_one(
    rows: list[tuple[np.ndarray, np.ndarray]],
    query_location: int,
    item_locations: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    """Ранжирует объединение кандидатов одного запроса.

    Args:
        rows: Top-500 и score трёх каналов.
        query_location: Идентификатор локации запроса.
        item_locations: Локации строк корпуса.
        item_ids: Идентификаторы строк корпуса для стабильных ничьих.

    Returns:
        До 50 номеров строк корпуса.
    """
    nonempty = [indices for indices, _ in rows if len(indices)]
    if not nonempty:
        return np.empty(0, dtype=np.int32)
    union = np.unique(np.concatenate(nonempty)).astype(np.int32, copy=False)
    scores = np.zeros(len(union), dtype=np.float32)
    for indices, raw_scores in rows:
        positions = np.searchsorted(union, indices)
        scores[positions] += minmax(raw_scores)
    scores /= len(rows)
    scores += (
        item_locations[union] == query_location
    ).astype(np.float32) * LOCATION_BONUS

    if len(union) > TOP_K:
        threshold = np.partition(scores, -TOP_K)[-TOP_K]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        remaining = TOP_K - len(above)
        tied_order = np.argsort(item_ids[union[tied]], kind="stable")[:remaining]
        selected = np.concatenate([above, tied[tied_order]])
        union, scores = union[selected], scores[selected]
    order = np.lexsort((item_ids[union], -scores))
    return union[order]


def fuse_all(
    channels: list[CandidateBatch],
    queries: pd.DataFrame,
    item_locations: np.ndarray,
    item_ids: np.ndarray,
) -> list[np.ndarray]:
    """Применяет mean-minmax fusion ко всем benchmark-запросам.

    Args:
        channels: Word BM25, fastText и char BM25 в этом порядке.
        queries: Benchmark-запросы.
        item_locations: Локации строк корпуса.
        item_ids: Идентификаторы строк корпуса.

    Returns:
        Номера итоговых объявлений для каждого запроса.

    Raises:
        ValueError: Если передано не три retrieval-канала.
    """
    if len(channels) != 3:
        raise ValueError("Для fusion нужны ровно три retrieval-канала")
    predictions: list[np.ndarray] = []
    for number, location in enumerate(queries["search_location_id"]):
        rows = [channel.row(number) for channel in channels]
        predictions.append(
            fuse_one(rows, int(location), item_locations, item_ids)
        )
    return predictions
