"""Формирует и строго проверяет submission задачи.

Проверки повторяют требования ``TASK.md`` и выполняются до записи результата,
а также доступны через отдельный entry point ``validate.py``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd

from settings import TOP_K


ITEM_ID_RE = re.compile(r"[0-9a-f]{16}")


def build_submission(
    queries: pd.DataFrame,
    predictions: Sequence[np.ndarray],
    item_ids: np.ndarray,
) -> pd.DataFrame:
    """Преобразует номера строк корпуса в требуемый CSV-формат.

    Args:
        queries: Benchmark-запросы в исходном порядке.
        predictions: Номера найденных строк для каждого запроса.
        item_ids: Item ID по строкам корпуса.

    Returns:
        Таблицу с колонками ``query_id`` и ``answer``.

    Raises:
        ValueError: Если число списков не совпадает с числом запросов.
    """
    if len(predictions) != len(queries):
        raise ValueError("Число предсказаний не совпадает с числом запросов")
    answers = [" ".join(item_ids[indices].tolist()) for indices in predictions]
    return pd.DataFrame(
        {"query_id": queries["query_id"].astype(str), "answer": answers}
    )


def validate_submission(
    answer: pd.DataFrame,
    queries: pd.DataFrame,
    valid_item_ids: Collection[str],
) -> None:
    """Проверяет все инварианты ``answer.csv`` из постановки.

    Args:
        answer: Загруженный или только что построенный submission.
        queries: Исходная таблица benchmark-запросов.
        valid_item_ids: Полный набор допустимых item ID.

    Raises:
        ValueError: Если колонки, запросы или ответы имеют неверный формат.
    """
    if list(answer.columns) != ["query_id", "answer"]:
        raise ValueError("Ожидаются ровно колонки query_id, answer")
    query_ids = queries["query_id"].astype(str)
    actual_query_ids = answer["query_id"].astype(str)
    if answer["query_id"].isna().any() or answer["answer"].isna().any():
        raise ValueError("В submission есть пропущенные значения")
    if not actual_query_ids.str.len().eq(16).all():
        raise ValueError("Каждый query_id должен состоять ровно из 16 символов")
    if actual_query_ids.duplicated().any():
        raise ValueError("В submission повторяются query_id")
    if len(answer) != len(query_ids) or set(actual_query_ids) != set(query_ids):
        raise ValueError("Набор query_id не совпадает с benchmark")

    allowed = set(valid_item_ids)
    for query_id, value in zip(actual_query_ids, answer["answer"].astype(str)):
        ids = value.split()
        if not 1 <= len(ids) <= TOP_K:
            raise ValueError(f"У {query_id} должно быть от 1 до {TOP_K} item_id")
        if len(ids) != len(set(ids)):
            raise ValueError(f"У {query_id} повторяются item_id")
        if any(ITEM_ID_RE.fullmatch(item_id) is None for item_id in ids):
            raise ValueError(f"У {query_id} есть item_id неверного формата")
        if any(item_id not in allowed for item_id in ids):
            raise ValueError(f"У {query_id} есть item_id вне benchmark-корпуса")
