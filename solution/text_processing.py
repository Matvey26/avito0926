"""Готовит тексты для word BM25 и char BM25.

Word-ветка применяет русскую лемматизацию, char-ветка — только нижний регистр,
замену ``ё`` и нормализацию пробелов. Поля не повторяются.
"""

from __future__ import annotations

import re
import time

import pandas as pd
import pymorphy3


TOKEN_RE = re.compile(r"(?u)\b\w{2,}\b")
RUSSIAN_TOKEN_RE = re.compile(r"^[а-я]+$")


def normalize_text(text: object) -> str:
    """Применяет лёгкую детерминированную нормализацию.

    Args:
        text: Текстовый скаляр или пропуск.

    Returns:
        Нормализованную строку либо пустую строку для пропуска.
    """
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


class RussianLemmatizer:
    """Лемматизирует русские токены и кеширует результат."""

    def __init__(self) -> None:
        """Создаёт Pymorphy3-анализатор и пустой кеш."""
        self.morph = pymorphy3.MorphAnalyzer()
        self.cache: dict[str, str] = {}

    def __call__(self, text: object) -> str:
        """Токенизирует и лемматизирует текст.

        Args:
            text: Значение, принимаемое :func:`normalize_text`.

        Returns:
            Токены длиной от двух символов через пробел.
        """
        tokens = TOKEN_RE.findall(normalize_text(text))
        for token in dict.fromkeys(tokens):
            if token in self.cache:
                continue
            if RUSSIAN_TOKEN_RE.fullmatch(token):
                lemma = self.morph.parse(token)[0].normal_form.replace("ё", "е")
            else:
                lemma = token
            self.cache[token] = lemma
        return " ".join(self.cache[token] for token in tokens)


def process(values: pd.Series, processor: object, label: str) -> pd.Series:
    """Обрабатывает Series порциями и показывает прогресс.

    Args:
        values: Исходные тексты.
        processor: Вызываемый обработчик одного текста.
        label: Название этапа.

    Returns:
        Подготовленные строки в исходном порядке.
    """
    started = time.monotonic()
    result: list[str] = []
    for start in range(0, len(values), 25_000):
        stop = min(start + 25_000, len(values))
        result.extend(processor(value) for value in values.iloc[start:stop])
        print(f"  {label}: {stop:,}/{len(values):,}", flush=True)
    print(f"  {label}: {time.monotonic() - started:.1f} с", flush=True)
    return pd.Series(result, index=values.index, dtype="object")


def prepare_word_documents(
    items: pd.DataFrame, queries: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Строит лемматизированные документы word BM25.

    Args:
        items: Benchmark-корпус объявлений.
        queries: Benchmark-запросы.

    Returns:
        Полные тексты объявлений и запросы с параметрами.
    """
    lemma = RussianLemmatizer()
    title = process(items["item_title_raw"], lemma, "word: заголовки")
    params = process(items["item_infm_params_text"], lemma, "word: параметры")
    description = process(items["item_description_raw"], lemma, "word: описания")
    query = process(queries["search_query"], lemma, "word: запросы")
    query_params = process(
        queries["search_infm_params_text"], lemma, "word: параметры запросов"
    )
    return title + " " + params + " " + description, query + " " + query_params


def prepare_char_documents(
    items: pd.DataFrame, queries: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Строит plain-документы char BM25.

    Args:
        items: Benchmark-корпус объявлений.
        queries: Benchmark-запросы.

    Returns:
        Полные тексты объявлений и запросы с параметрами.
    """
    title = items["item_title_raw"].map(normalize_text)
    params = items["item_infm_params_text"].map(normalize_text)
    description = items["item_description_raw"].map(normalize_text)
    query = queries["search_query"].map(normalize_text)
    query_params = queries["search_infm_params_text"].map(normalize_text)
    return title + " " + params + " " + description, query + " " + query_params
