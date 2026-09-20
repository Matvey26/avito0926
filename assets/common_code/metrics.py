"""Считает полноту поиска и агрегирует результаты cold/warm-валидации.

Функции работают с уже сформированными списками кандидатов и реализуют единый
для проекта расчёт метрик. Здесь же строится фиксированная смесь cold/warm в
пропорции 63/37, используемая для оценки ожидаемого качества на бенчмарке.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Hashable

import numpy as np
import pandas as pd


DEFAULT_MODE_WEIGHTS = {"cold": 0.63, "warm": 0.37}
DEFAULT_MIXTURE_COLUMN = "benchmark_mix_63_cold_37_warm"


def count_relevant(
    predicted: Iterable[Hashable], relevant: Collection[Hashable]
) -> int:
    """Считает предсказанных кандидатов, входящих в множество релевантных.

    Аргументы:
        predicted: Найденные идентификаторы или номера строк корпуса. Ожидается
            уже уникальная последовательность, обрезанная до top-k.
        relevant: Коллекция релевантных идентификаторов для того же запроса.

    Возвращает:
        Количество значений из ``predicted``, содержащихся в ``relevant``.
    """
    return sum(candidate in relevant for candidate in predicted)


def recall_at_k(
    predicted: Iterable[Hashable], relevant: Collection[Hashable]
) -> float:
    """Вычисляет recall одного запроса для уже обрезанного списка кандидатов.

    Аргументы:
        predicted: Уникальные идентификаторы кандидатов для одного запроса.
        relevant: Эталонные релевантные идентификаторы этого запроса.

    Возвращает:
        Долю релевантных идентификаторов, присутствующих в ``predicted``.

    Исключения:
        ValueError: Если ``relevant`` пуст, поскольку recall не определён.
    """
    if not relevant:
        raise ValueError("Recall is undefined for a query without relevant items")
    return count_relevant(predicted, relevant) / len(relevant)


def summarize_recalls(recalls: Sequence[float]) -> dict[str, object]:
    """Агрегирует значения recall отдельных запросов одного сплита.

    Аргументы:
        recalls: Значения recall по запросам, обычно с отсечкой 50.

    Возвращает:
        Словарь с числом запросов, макроусреднённым Recall@50 и количеством
        запросов хотя бы с одним попаданием.

    Примечания:
        Для пустой последовательности NumPy вернёт ``NaN`` в поле
        ``recall_at_50``; в штатном сценарии сплиты непустые.
    """
    return {
        "queries": len(recalls),
        "recall_at_50": float(np.mean(recalls)),
        "queries_with_hit": int(np.count_nonzero(recalls)),
    }


def add_mode_mixture(
    metrics: pd.DataFrame,
    config_columns: Sequence[str],
    mode_weights: dict[str, float] = DEFAULT_MODE_WEIGHTS,
    metric_column: str = "recall_at_50",
    mixture_column: str = DEFAULT_MIXTURE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Добавляет взвешенную смесь сплитов и строит широкую таблицу сравнения.

    Аргументы:
        metrics: Длинная таблица метрик с колонками конфигурации, ``mode`` и
            выбранной метрикой.
        config_columns: Колонки, однозначно определяющие конфигурацию.
        mode_weights: Соответствие режима валидации и его веса в итоговой
            смеси. По умолчанию используются 63% cold и 37% warm.
        metric_column: Название метрики для разворота и смешивания.
        mixture_column: Название колонки со взвешенной оценкой.

    Возвращает:
        Пару ``(enriched, comparison)``. ``enriched`` сохраняет исходный
        длинный формат и добавляет смесь, а ``comparison`` содержит одну строку
        на конфигурацию и отдельную колонку метрики для каждого режима.

    Исключения:
        KeyError: Если отсутствуют необходимые колонки конфигурации, режима или
            метрики либо недоступен режим из ``mode_weights``.
    """
    columns = list(config_columns)
    comparison = metrics.pivot_table(
        index=columns,
        columns="mode",
        values=metric_column,
    ).reset_index()
    comparison[mixture_column] = sum(
        weight * comparison[mode] for mode, weight in mode_weights.items()
    )
    enriched = metrics.merge(
        comparison[columns + [mixture_column]],
        on=columns,
        how="left",
    )
    return enriched, comparison
