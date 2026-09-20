"""Строит детерминированные cold/warm-выборки и множества релевантности.

Методы поиска оцениваются на стабильных прокси-сплитах из положительных
обучающих пар. Отбор по хешу не зависит от поведения генератора случайных
чисел, а полные сигнатуры сохраняют все признаки запроса, нужные пайплайну.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import pandas as pd


VALIDATION_SEED = "20260918"
QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


def stable_hash(value: str, seed: str = VALIDATION_SEED) -> str:
    """Возвращает детерминированный SHA-1 для упорядочивания выборки.

    Аргументы:
        value: Текстовое представление упорядочиваемого объекта.
        seed: Зерно протокола, добавляемое перед ``value`` до хеширования.

    Возвращает:
        SHA-1 в виде шестнадцатеричной строки нижнего регистра, пригодной как
        стабильный ключ сортировки.
    """
    return hashlib.sha1(f"{seed}|{value}".encode()).hexdigest()


def signature_key(
    row: pd.Series, query_columns: Sequence[str] = QUERY_COLUMNS
) -> tuple[object, ...]:
    """Преобразует строку запроса в канонический ключ релевантности.

    Аргументы:
        row: Строка pandas со всеми необходимыми колонками запроса.
        query_columns: Упорядоченные колонки полной сигнатуры запроса.

    Возвращает:
        Кортеж значений в точном порядке ``query_columns``.

    Исключения:
        KeyError: Если в ``row`` отсутствует хотя бы одна требуемая колонка.
    """
    return tuple(row[column] for column in query_columns)


def build_validation(
    train: pd.DataFrame,
    mode: str,
    size: int,
    normalize_text: Callable[[object], str],
    query_columns: Sequence[str] = QUERY_COLUMNS,
    seed: str = VALIDATION_SEED,
) -> tuple[pd.DataFrame, dict[tuple[object, ...], set[str]]]:
    """Строит одну детерминированную cold- или warm-выборку.

    Для каждого нормализованного текста выбирается одна полная сигнатура.
    Cold-режим выбирает из всех текстов, а warm — только из текстов минимум с
    двумя различными полными сигнатурами. Сортировка по хешу делает отбор
    текстов и представительных сигнатур воспроизводимым.

    Аргументы:
        train: Положительные пары «запрос — объявление» с колонками запроса и
            ``item_id``.
        mode: Режим ``cold`` или ``warm``.
        size: Максимальное число выбираемых нормализованных текстов запроса.
        normalize_text: Функция группировки эквивалентных написаний запроса.
        query_columns: Упорядоченные колонки полной сигнатуры запроса.
        seed: Зерно стабильного хеша, управляющее отбором.

    Возвращает:
        DataFrame с одной строкой на выбранную сигнатуру и соответствие каждой
        сигнатуры множеству релевантных ``item_id``.

    Исключения:
        KeyError: Если в ``train`` отсутствуют необходимые колонки.
        ValueError: Если ``mode`` не равен ``cold`` или ``warm``.
    """
    columns = list(query_columns)
    grouped = train.groupby(columns, sort=False, dropna=False)["item_id"].agg(
        lambda values: set(values)
    )
    signatures_by_text: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for key in grouped.index:
        signatures_by_text[normalize_text(key[0])].append(key)

    if mode == "cold":
        candidate_texts = list(signatures_by_text)
    elif mode == "warm":
        candidate_texts = [
            text
            for text, signatures in signatures_by_text.items()
            if len(signatures) >= 2
        ]
    else:
        raise ValueError(f"Unknown validation mode: {mode}")

    hash_value = lambda value: stable_hash(value, seed)
    selected_texts = sorted(candidate_texts, key=hash_value)[:size]
    selected_keys = [
        min(
            signatures_by_text[text],
            key=lambda key: hash_value(repr(tuple(str(value) for value in key))),
        )
        for text in selected_texts
    ]
    validation = pd.DataFrame(selected_keys, columns=columns)
    relevance = {key: grouped.loc[key] for key in selected_keys}
    return validation, relevance


def build_validation_splits(
    train: pd.DataFrame,
    size: int,
    normalize_text: Callable[[object], str],
    modes: Iterable[str] = ("cold", "warm"),
    query_columns: Sequence[str] = QUERY_COLUMNS,
    seed: str = VALIDATION_SEED,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[tuple[object, ...], set[str]]],
]:
    """Строит несколько режимов валидации по единому протоколу.

    Аргументы:
        train: Положительные обучающие пары «запрос — объявление».
        size: Максимальное число текстов запроса в каждом сплите.
        normalize_text: Функция нормализации текста запроса.
        modes: Создаваемые режимы, обычно ``cold`` и ``warm``.
        query_columns: Упорядоченные колонки полной сигнатуры запроса.
        seed: Общее для всех режимов зерно стабильного хеша.

    Возвращает:
        Два словаря по режимам: валидационные DataFrame и соответствия
        сигнатур множествам релевантных объявлений.

    Исключения:
        ValueError: Если запрошен неподдерживаемый режим.
    """
    validations: dict[str, pd.DataFrame] = {}
    relevances: dict[str, dict[tuple[object, ...], set[str]]] = {}
    for mode in modes:
        validations[mode], relevances[mode] = build_validation(
            train,
            mode,
            size,
            normalize_text,
            query_columns=query_columns,
            seed=seed,
        )
    return validations, relevances


def relevance_ids_to_rows(
    relevance_ids: dict[tuple[object, ...], set[str]],
    item_row_by_id: dict[str, int],
) -> dict[tuple[object, ...], frozenset[int]]:
    """Заменяет релевантные item ID компактными номерами строк корпуса.

    Аргументы:
        relevance_ids: Релевантные item ID по полным сигнатурам запросов.
        item_row_by_id: Соответствие каждого item ID строке корпуса.

    Возвращает:
        Соответствие релевантности с неизменяемыми множествами номеров строк.

    Исключения:
        KeyError: Если релевантного item ID нет в ``item_row_by_id``.
    """
    return {
        key: frozenset(item_row_by_id[item_id] for item_id in item_ids)
        for key, item_ids in relevance_ids.items()
    }


def target_category(value: object) -> int:
    """Определяет категорию для ограниченного по категории поиска.

    Аргументы:
        value: Значение категории запроса, приводимое к целому числу.

    Возвращает:
        Категорию 114 для нулевой категории запроса, иначе исходную категорию.
        EDA показал, что у всех наблюдавшихся запросов категории 0 положительные
        объявления относились к категории 114.

    Исключения:
        TypeError: Если ``value`` нельзя интерпретировать как скалярное целое.
        ValueError: Если приведение к целому не удалось.
    """
    category = int(value)
    return 114 if category == 0 else category
